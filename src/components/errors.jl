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
    rel_primal = round(primal / primal_scale; sigdigits = 3)
    rel_dual = round(dual / dual_scale; sigdigits = 3)
    rel_gap = round(gap / gap_scale; sigdigits = 3)
    return print(
        io, """KKT relative errors: primal $rel_primal, dual $rel_dual, gap $rel_gap"""
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
    lcuc2 = @. scratch.y = squared_bound_scale(lc, uc)
    primal_scale = one(T) + sqrt(sum(lcuc2))

    dual_diff = @. scratch.x = inv(D2.diag) * (c - At_y - r)
    dual = norm(dual_diff)
    dual_scale = one(T) + norm(c)

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
