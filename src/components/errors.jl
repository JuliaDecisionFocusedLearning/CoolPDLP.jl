"""
    KKTErrors

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct KKTErrors{T <: BatchedNumber}
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

"""
    batch_select(cond, err1, err2)

Take the errors of `err1` where `cond` holds and those of `err2` elsewhere, column by column.
"""
function batch_select(cond, err1::KKTErrors, err2::KKTErrors)
    pick(e1, e2) = ifelse.(cond, e1, e2)
    return KKTErrors(
        pick(err1.primal, err2.primal),
        pick(err1.primal_scale, err2.primal_scale),
        pick(err1.dual, err2.dual),
        pick(err1.dual_scale, err2.dual_scale),
        pick(err1.gap, err2.gap),
        pick(err1.gap_scale, err2.gap_scale),
    )
end

function relative(err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return @. max(primal / primal_scale, dual / dual_scale, gap / gap_scale)
end

function absolute(err::KKTErrors, ω::BatchedNumber)
    (; primal, dual, gap) = err
    return @. sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)
end

function kkt_errors!(
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
    primal = colnorm(primal_diff)

    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc)
    primal_scale = one(T) .+ colnorm(rescaled_combined_bounds)

    dual_diff = @. scratch.x = inv(D2.diag) * (c_At_y - z)
    dual = colnorm(dual_diff)

    rescaled_obj = @. scratch.x = inv(D2.diag) * c
    dual_scale = one(T) .+ colnorm(rescaled_obj)

    # dual objective:   lᵀ|y|⁺ - uᵀ|y|⁻ + lᵥᵀ|z|⁺ - uᵥᵀ|z|⁻
    #    We reformulate to ∑ⱼ (l⋅|y|⁺ - u⋅|y|⁻)ⱼ + ∑ᵢ (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)ᵢ
    #    where pc = (l⋅|y|⁺ - u⋅|y|⁻) and pv = (lᵥ⋅|z|⁺ - uᵥ⋅|z|⁻)
    pc = @. scratch.y = (
        safeprod_left(lc, positive_part(y)) - safeprod_left(uc, negative_part(y))
    )
    pv = @. scratch.z = (
        safeprod_left(lv, positive_part(z)) - safeprod_left(uv, negative_part(z))
    )
    pc_sum = colsum(pc)
    pv_sum = colsum(pv)
    cx = colsum(@. scratch.x = c * x)
    dobj = @. pc_sum + pv_sum

    gap = @. abs(cx - dobj)
    gap_scale = @. one(T) + abs(dobj) + abs(cx)

    err = KKTErrors(;
        primal,
        dual,
        gap,
        primal_scale,
        dual_scale,
        gap_scale,
    )
    return err
end
