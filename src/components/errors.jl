"""
    KKTErrors

Mutable so that [`kkt_errors!`](@ref) can refill it without allocating.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct KKTErrors{T <: Number, B <: BatchedNumber{T}}
    "primal feasibility error"
    primal::B
    "characteristic scale of the primal constraint RHS"
    primal_scale::B
    "dual feasibility error"
    dual::B
    "characteristic scale of the dual constraint RHS"
    dual_scale::B
    "primal-dual gap"
    gap::B
    "characteristic scale of the gap"
    gap_scale::B
end

format_error(e::Number) = @sprintf("%.3e", e)
function format_error(e::AbstractVector)
    return "[" * join((@sprintf("%.3e", eᵢ) for eᵢ in adapt(CPU(), e)), ", ") * "]"
end

function Base.show(io::IO, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    rel_primal = format_error(primal ./ primal_scale)
    rel_dual = format_error(dual ./ dual_scale)
    rel_gap = format_error(gap ./ gap_scale)
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

function KKTErrors(sol::PrimalDualSolution{T}) where {T}
    nan() = batched_expand(sol.x, convert(T, NaN))
    return KKTErrors(nan(), nan(), nan(), nan(), nan(), nan())
end

instance(err::KKTErrors, i::Int) = KKTErrors(
    instance_num(err.primal, i),
    instance_num(err.primal_scale, i),
    instance_num(err.dual, i),
    instance_num(err.dual_scale, i),
    instance_num(err.gap, i),
    instance_num(err.gap_scale, i),
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
    select_errors!!(dest, cond, err_true, err_false)

Fill `dest`, column by column, with the errors of `err_true` where `cond` holds and those of `err_false` elsewhere.
"""
function select_errors!!(
        dest::KKTErrors, cond::BatchedNumber,
        err_true::KKTErrors, err_false::KKTErrors,
    )
    dest.primal = broadcast!!(ifelse, dest.primal, cond, err_true.primal, err_false.primal)
    dest.primal_scale = broadcast!!(
        ifelse, dest.primal_scale, cond, err_true.primal_scale, err_false.primal_scale
    )
    dest.dual = broadcast!!(ifelse, dest.dual, cond, err_true.dual, err_false.dual)
    dest.dual_scale = broadcast!!(
        ifelse, dest.dual_scale, cond, err_true.dual_scale, err_false.dual_scale
    )
    dest.gap = broadcast!!(ifelse, dest.gap, cond, err_true.gap, err_false.gap)
    dest.gap_scale = broadcast!!(
        ifelse, dest.gap_scale, cond, err_true.gap_scale, err_false.gap_scale
    )
    return dest
end

"""
    relative!!(dest, err)

Compute the largest relative KKT error, column by column, into `dest`.
"""
function relative!!(dest::BatchedNumber, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return broadcast!!(
        dest, primal, primal_scale, dual, dual_scale, gap, gap_scale
    ) do p, ps, d, ds, g, gs
        max(p / ps, d / ds, g / gs)
    end
end

"""
    relative(err)

Compute the largest relative KKT error, column by column.
"""
relative(err::KKTErrors) = relative!!(batched_similar(err.primal), err)

"""
    absolute!!(dest, err, ω)

Compute the absolute KKT error for primal weight `ω`, column by column, into `dest`.
"""
function absolute!!(dest::BatchedNumber, err::KKTErrors, ω::BatchedNumber)
    (; primal, dual, gap) = err
    return broadcast!!(dest, primal, dual, gap, ω) do p, d, g, w
        sqrt(w^2 * p^2 + inv(w^2) * d^2 + g^2)
    end
end

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
    err.primal = colnorm!!(err.primal, primal_diff)

    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc)
    err.primal_scale = colnorm!!(err.primal_scale, rescaled_combined_bounds)
    err.primal_scale = broadcast!!(+, err.primal_scale, one(T), err.primal_scale)

    dual_diff = @. scratch.x = inv(D2.diag) * (c_At_y - z)
    err.dual = colnorm!!(err.dual, dual_diff)

    rescaled_obj = @. scratch.x = inv(D2.diag) * c
    err.dual_scale = colnorm!!(err.dual_scale, rescaled_obj)
    err.dual_scale = broadcast!!(+, err.dual_scale, one(T), err.dual_scale)

    # dual objective:   lᵀ|y|⁺ - uᵀ|y|⁻ + lᵥᵀ|z|⁺ - uᵥᵀ|z|⁻
    #    We reformulate to ∑ⱼ (l⋅|y|⁺ - u⋅|y|⁻)ⱼ + ∑ᵢ (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)ᵢ
    #    where pc = (l⋅|y|⁺ - u⋅|y|⁻) and pv = (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)
    pc = @. scratch.y = (
        safeprod_left(lc, positive_part(y)) - safeprod_left(uc, negative_part(y))
    )
    pv = @. scratch.z = (
        safeprod_left(lv, positive_part(z)) - safeprod_left(uv, negative_part(z))
    )
    pc_sum = colsum!!(scratch.b1, pc)
    pv_sum = colsum!!(scratch.b2, pv)
    dobj = broadcast!!(+, scratch.b1, pc_sum, pv_sum)
    cx = colsum!!(scratch.b2, @. scratch.x = c * x)

    err.gap = broadcast!!((a, b) -> abs(a - b), err.gap, cx, dobj)
    err.gap_scale = broadcast!!((a, b) -> one(T) + abs(a) + abs(b), err.gap_scale, dobj, cx)
    return err
end
