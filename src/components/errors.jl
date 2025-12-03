"""
    KKTErrors

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct KKTErrors{T <: Number}
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

function Base.show(io::IO, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    rel_primal = @sprintf("%.3e", primal / primal_scale)
    rel_dual = @sprintf("%.3e", dual / dual_scale)
    rel_gap = @sprintf("%.3e", gap / gap_scale)
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

function relative(err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return max(primal / primal_scale, dual / dual_scale, gap / gap_scale)
end

function absolute(err::KKTErrors, ω::Number)
    (; primal, dual, gap) = err
    return sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)
end

function kkt_errors!(
        scratch::Scratch,
        sol::PrimalDualSolution,
        milp::MILP{T},
    ) where {T}
    (; x, y) = sol
    (; c, lv, uv, A, At, lc, uc, D1, D2) = milp

    A_x = mul!(scratch.y, A, x)
    At_y = mul!(scratch.x, At, y)
    r = @. scratch.r = proj_multiplier(c - At_y, lv, uv)

    primal_diff = @. scratch.y = inv(D1.diag) * (A_x - proj_box(A_x, lc, uc))
    primal = norm(primal_diff)
    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc)
    primal_scale = one(T) + norm(rescaled_combined_bounds)

    dual_diff = @. scratch.x = inv(D2.diag) * (c - At_y - r)
    dual = norm(dual_diff)
    rescaled_obj = @. scratch.x = inv(D2.diag) * c
    dual_scale = one(T) + norm(rescaled_obj)

    pc = @. scratch.y = (
        safeprod_left(uc, positive_part(-y)) - safeprod_left(lc, negative_part(-y))
    )
    pv = @. scratch.r = (
        safeprod_left(uv, positive_part(-r)) - safeprod_left(lv, negative_part(-r))
    )
    pc_sum = sum(pc)
    pv_sum = sum(pv)
    cx = dot(c, x)

    gap = abs(cx + pc_sum + pv_sum)
    gap_scale = one(T) + abs(pc_sum + pv_sum) + abs(cx)

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
