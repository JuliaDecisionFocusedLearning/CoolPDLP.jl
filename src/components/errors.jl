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

_half_xQx(cx, x, milp::LinearProgram{T}) where {T} = zero(T)
_half_xQx(cx, x, milp::QuadraticProgram) = (cx - dot(milp.c, x)) / 2

grad_x(scratch, x, milp::LinearProgram) = milp.c
function grad_x(scratch, x, milp::QuadraticProgram)
    mul!(scratch.r, milp.Q, x)
    @. scratch.r += milp.c
    return scratch.r
end

function kkt_errors!(
        scratch::Scratch,
        sol::PrimalDualSolution,
        milp::AbstractProgram{T},
    ) where {T}
    (; x, y) = sol
    (; lv, uv, A, At, lc, uc, D1, D2) = milp
    g = grad_x(scratch, x, milp)

    cx = dot(g, x)
    rescaled_obj = @. scratch.x = inv(D2.diag) * g
    dual_scale = one(T) + norm(rescaled_obj)

    A_x = mul!(scratch.y, A, x)
    At_y = mul!(scratch.x, At, y)

    h = @. scratch.r = g - At_y
    dual_diff = @. scratch.x = inv(D2.diag) * (h - proj_multiplier(h, lv, uv))
    dual = norm(dual_diff)
    # h is no longer needed, reuse the scratch.r
    r = @. scratch.r = proj_multiplier(scratch.r, lv, uv)

    primal_diff = @. scratch.y = inv(D1.diag) * (A_x - proj_box(A_x, lc, uc))
    primal = norm(primal_diff)
    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc)
    primal_scale = one(T) + norm(rescaled_combined_bounds)

    # primal obj P = cᵀx + ½xᵀQx
    # dual obj   D = lᵀy⁺ − uᵀy⁻ + lᵥᵀr⁺ − uᵥᵀr⁻ − ½xᵀQx
    # gap = |P − D|
    #     = |(cᵀx + ½xᵀQx) − (lᵀy⁺ − uᵀy⁻ + lᵥᵀr⁺ − uᵥᵀr⁻ − ½xᵀQx)|
    #     = |cᵀx + xᵀQx + (uᵀy⁻ − lᵀy⁺) + (uᵥᵀr⁻ − lᵥᵀr⁺)|
    #     = |gᵀx + pc_sum + pv_sum|;  g = c + Qx
    # gap_scale = 1 + |P| + |D| = 1 + |gᵀx − ½xᵀQx| + |pc_sum + pv_sum + ½xᵀQx|
    pc = @. scratch.y = (
        safeprod_left(uc, positive_part(-y)) - safeprod_left(lc, negative_part(-y))
    )
    pv = @. scratch.r = (
        safeprod_left(uv, positive_part(-r)) - safeprod_left(lv, negative_part(-r))
    )
    pc_sum = sum(pc)
    pv_sum = sum(pv)

    gap = abs(cx + pc_sum + pv_sum)
    half_xQx = _half_xQx(cx, x, milp)
    gap_scale = one(T) + abs(cx - half_xQx) + abs(pc_sum + pv_sum + half_xQx)

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
