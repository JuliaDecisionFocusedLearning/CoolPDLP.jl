"""
    KKTErrors

Mutable so that [`kkt_errors!`](@ref) can refill it without allocating.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct KKTErrors{T <: BatchedNumber}
    "primal feasibility error"
    primal::T
    "characteristic scale of the primal constraint RHS"
    primal_scale::T
    "dual feasibility error"
    dual::T
    "characteristic scale of the dual constraint RHS"
    dual_scale::T
    "primal-dual gap"
    gap::T
    "characteristic scale of the gap"
    gap_scale::T
end

format_error(e::Number) = @sprintf("%.3e", e)
function format_error(e::AbstractVector)
    return "[" * join((@sprintf("%.3e", eᵢ) for eᵢ in adapt(CPU(), e)), ", ") * "]"
end

function Base.show(io::IO, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    rel_primal = format_error(@. primal / primal_scale)
    rel_dual = format_error(@. dual / dual_scale)
    rel_gap = format_error(@. gap / gap_scale)
    return print(
        io, """KKT relative errors: primal $rel_primal, dual $rel_dual, gap $rel_gap"""
    )
end

function Base.isapprox(err1::KKTErrors, err2::KKTErrors; kwargs...)
    return (
        isapprox(err1.primal, err2.primal; kwargs...) &&
            isapprox(err1.dual, err2.dual; kwargs...) &&
            isapprox(err1.gap, err2.gap; kwargs...) &&
            isapprox(err1.primal_scale, err2.primal_scale; kwargs...) &&
            isapprox(err1.dual_scale, err2.dual_scale; kwargs...) &&
            isapprox(err1.gap_scale, err2.gap_scale; kwargs...)
    )
end

function KKTErrors(::Type{T}) where {T}
    return KKTErrors(
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
    )
end

function KKTErrors(sol::PrimalDualSolution{T}) where {T}
    nan() = batch_expand(sol.x, convert(T, NaN))
    return KKTErrors(nan(), nan(), nan(), nan(), nan(), nan())
end

batch(err::KKTErrors, i::Int) = KKTErrors(
    batch_num(err.primal, i),
    batch_num(err.primal_scale, i),
    batch_num(err.dual, i),
    batch_num(err.dual_scale, i),
    batch_num(err.gap, i),
    batch_num(err.gap_scale, i),
)

Base.copy(err::KKTErrors) = KKTErrors(
    copy(err.primal),
    copy(err.primal_scale),
    copy(err.dual),
    copy(err.dual_scale),
    copy(err.gap),
    copy(err.gap_scale),
)

"""
    batch_select!(err, cond, other)

Replace the errors of `err` by those of `other` in the columns where `cond` holds.
"""
function batch_select!(err::KKTErrors, cond, other::KKTErrors)
    pick(e, o) = batch_apply!(ifelse, e, cond, o, e)
    err.primal = pick(err.primal, other.primal)
    err.primal_scale = pick(err.primal_scale, other.primal_scale)
    err.dual = pick(err.dual, other.dual)
    err.dual_scale = pick(err.dual_scale, other.dual_scale)
    err.gap = pick(err.gap, other.gap)
    err.gap_scale = pick(err.gap_scale, other.gap_scale)
    return err
end

"""
    relative!(dest, err)

Compute the largest relative KKT error, column by column, into `dest`.
"""
function relative!(::Number, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return max(primal / primal_scale, dual / dual_scale, gap / gap_scale)
end

function relative!(dest::AbstractVector, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    @. dest = max(primal / primal_scale, dual / dual_scale, gap / gap_scale)
    return dest
end

relative(err::KKTErrors) = relative!(batch_similar(err.primal), err)

"""
    absolute!(dest, err, ω)

Compute the absolute KKT error for primal weight `ω`, column by column, into `dest`.
"""
function absolute!(::Number, err::KKTErrors, ω::BatchedNumber)
    (; primal, dual, gap) = err
    return sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)
end

function absolute!(dest::AbstractVector, err::KKTErrors, ω::BatchedNumber)
    (; primal, dual, gap) = err
    @. dest = sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)
    return dest
end

absolute(err::KKTErrors, ω::BatchedNumber) = absolute!(batch_similar(err.primal), err, ω)

"""
    kkt_errors!(err, scratch, sol, milp)

Fill `err` with the KKT errors of `sol`, one value per column of the batch.
"""
function kkt_errors!(
        err::KKTErrors,
        scratch::Scratch,
        sol::PrimalDualSolution,
        milp::MILP{T},
    ) where {T}
    (; x, y) = sol
    (; c, lv, uv, A, At, lc, uc, D1, D2) = milp

    A_x = mul!(scratch.y, A, x)
    c_At_y = mul!(scratch.x, At, y, -one(T), zero(T))
    c_At_y .+= c
    z = @. scratch.z = proj_multiplier(c_At_y, lv, uv)

    primal_diff = @. scratch.y = inv(D1.diag) * (A_x - clamp(A_x, lc, uc))
    err.primal = colnorm!(err.primal, scratch, primal_diff)

    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc)
    err.primal_scale = colnorm!(err.primal_scale, scratch, rescaled_combined_bounds)
    err.primal_scale = batch_apply!(+, err.primal_scale, one(T), err.primal_scale)

    dual_diff = @. scratch.x = inv(D2.diag) * (c_At_y - z)
    err.dual = colnorm!(err.dual, scratch, dual_diff)

    rescaled_obj = @. scratch.x = inv(D2.diag) * c
    err.dual_scale = colnorm!(err.dual_scale, scratch, rescaled_obj)
    err.dual_scale = batch_apply!(+, err.dual_scale, one(T), err.dual_scale)

    # dual objective:   lᵀ|y|⁺ - uᵀ|y|⁻ + lᵥᵀ|z|⁺ - uᵥᵀ|z|⁻
    #    We reformulate to ∑ⱼ (l⋅|y|⁺ - u⋅|y|⁻)ⱼ + ∑ᵢ (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)ᵢ
    #    where pc = (l⋅|y|⁺ - u⋅|y|⁻) and pv = (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)
    pc = @. scratch.y = (
        safeprod_left(lc, positive_part(y)) - safeprod_left(uc, negative_part(y))
    )
    pv = @. scratch.z = (
        safeprod_left(lv, positive_part(z)) - safeprod_left(uv, negative_part(z))
    )
    pc_sum = colsum!(scratch.b1, scratch, pc)
    pv_sum = colsum!(scratch.b2, scratch, pv)
    dobj = batch_apply!(+, scratch.b1, pc_sum, pv_sum)
    cx = colsum!(scratch.b2, scratch, @. scratch.x = c * x)

    err.gap = batch_apply!((a, b) -> abs(a - b), err.gap, cx, dobj)
    err.gap_scale = batch_apply!((a, b) -> one(T) + abs(a) + abs(b), err.gap_scale, dobj, cx)
    return err
end
