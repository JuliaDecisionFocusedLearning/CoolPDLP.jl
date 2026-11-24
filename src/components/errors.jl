"""
    KKTErrors

Mutable so that [`kkt_errors!`](@ref) can refill it without allocating.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct KKTErrors{B <: BatchedNumber}
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
    absolute!!(dest, err, ω, relative_kkt = Val(:pdlp))

Compute the restart metric for primal weight `ω`, column by column, into `dest`: the absolute
KKT error `√(ω² p² + d² / ω² + g²)` of PDLP by default, or the relative weighted KKT error
`max(ω p / pₛ, d / (ω dₛ), g / gₛ)` of PDQP (`relative_kkt_residual` in its
`saddle_point_gpu.jl`) with `Val(:pdqp)`.
"""
absolute!!(dest::BatchedNumber, err::KKTErrors, ω::BatchedNumber) = absolute!!(dest, err, ω, Val(:pdlp))
function absolute!!(dest::BatchedNumber, err::KKTErrors, ω::BatchedNumber, ::Val{:pdqp})
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return broadcast!!(dest, primal, primal_scale, dual, dual_scale, gap, gap_scale, ω) do p, ps, d, ds, g, gs, w
        max(w * p / ps, d / (w * ds), g / gs)
    end
end
function absolute!!(dest::BatchedNumber, err::KKTErrors, ω::BatchedNumber, ::Val{:pdlp})
    (; primal, dual, gap) = err
    return broadcast!!(dest, primal, dual, gap, ω) do p, d, g, w
        sqrt(w^2 * p^2 + inv(w^2) * d^2 + g^2)
    end
end

"""
    kkt_errors!(err, scratch, sol, milp, relative_kkt, norm)

Fill `err` with the KKT errors of `sol`, one value per column of the batch.

The residuals and the norms of the bounds and the objective are measured in the [`TerminationNorm`](@ref) `norm`.

The errors are those of the original problem, before preconditioning and bound and objective
rescaling. For a quadratic program, the reduced costs are `c + Q x - Aᵀ y`, the primal
objective gains `½ xᵀQx` and the dual objective loses it, as in PDQP and HPR-QP.

By default (`Val(:pdlp)`, also used when `relative_kkt` is omitted), the residuals are measured
in `norm` and compared with `1 + ‖combined bounds‖`, `1 + ‖c‖` and
`1 + |pobj| + |dobj|`, as in PDLP. With `Val(:pdqp)`, they follow the relative KKT error of PDQP
(Lu & Yang, 2023, Section 5) and HPR-QP (Section 4.1), whatever `norm`: infinity norms, compared
with `1 + max(‖b‖∞, ‖Ax‖∞)`, `1 + max(‖c‖∞, ‖Qx‖∞, ‖Aᵀy‖∞)` and `1 + max(|pobj|, |dobj|)`, where
`b` holds the finite constraint bounds. That costs one more product by `A` and by `Aᵀ`.
"""
function kkt_errors!(
        err::KKTErrors, scratch::Scratch, sol::PrimalDualSolution, milp::MILP, norm::TerminationNorm
    )
    return kkt_errors!(err, scratch, sol, milp, Val(:pdlp), norm)
end
function kkt_errors!(
        err::KKTErrors, scratch::Scratch, sol::PrimalDualSolution,
        z_given::Union{Nothing, AbstractVecOrMat}, milp::MILP, norm::TerminationNorm,
    )
    return kkt_errors!(err, scratch, sol, z_given, milp, Val(:pdlp), norm)
end

function kkt_errors!(
        err::KKTErrors,
        scratch::Scratch,
        sol::PrimalDualSolution,
        milp::MILP,
        relative_kkt::Val,
        norm::TerminationNorm,
    )
    return kkt_errors!(err, scratch, sol, nothing, milp, relative_kkt, norm)
end

"""
    kkt_errors!(err, scratch, sol, z, milp, relative_kkt, norm)

Same as `kkt_errors!(err, scratch, sol, milp, relative_kkt, norm)`, with the reduced costs (multipliers
of the variable bounds) given by `z` rather than chosen as the projection of `c - Aᵀy` onto the
multipliers that the bounds allow. `z === nothing` falls back to that projection.

[`HPRLP`](@ref) uses it with the `z̄` of its splitting, as HPR-LP does: the dual residual is then
`‖c - Aᵀy - z‖` and the dual objective uses `z`. `z` is only read.
"""
function kkt_errors!(
        err::KKTErrors,
        scratch::Scratch,
        sol::PrimalDualSolution,
        z_given::Union{Nothing, AbstractVecOrMat},
        milp::MILP{T},
        ::Val{:pdlp},
        norm::TerminationNorm,
    ) where {T}
    (; x, y) = sol
    (; c, c0, lv, uv, A, At, Q, lc, uc, D1, D2) = milp
    # the rescaled problem has bounds `β b`, objective `γ c`, quadratic term `(γ / β) Q` and
    # solution `(β x, γ y)`
    β, γ = milp.bound_scale, milp.obj_scale
    βr, γr = transpose(β), transpose(γ)

    A_x = mul!(scratch.y, A, x)
    # literals rather than `-one(T)` and `zero(T)`, which are traced under Reactant
    c_At_y = mul!(scratch.x, At, y, -1, false)
    # `err.gap` holds `xᵀQx` until the gap itself is computed
    xQx = add_quadratic_gradient!!(err.gap, scratch.z, c_At_y, Q, x)
    c_At_y .+= c
    z = bound_multipliers!(scratch.z, c_At_y, lv, uv, z_given)

    primal_diff = @. scratch.y = inv(D1.diag) * (A_x - clamp(A_x, lc, uc)) / βr
    err.primal = colnorm!!(err.primal, primal_diff, norm)

    rescaled_combined_bounds = @. scratch.y = inv(D1.diag) * combine(lc, uc) / βr
    err.primal_scale = colnorm!!(err.primal_scale, rescaled_combined_bounds, norm)
    err.primal_scale = broadcast!!(+, err.primal_scale, one(T), err.primal_scale)

    dual_diff = @. scratch.x = inv(D2.diag) * (c_At_y - z) / γr
    err.dual = colnorm!!(err.dual, dual_diff, norm)

    rescaled_obj = @. scratch.x = inv(D2.diag) * c / γr
    err.dual_scale = colnorm!!(err.dual_scale, rescaled_obj, norm)
    err.dual_scale = broadcast!!(+, err.dual_scale, one(T), err.dual_scale)

    # dual objective:   lᵀ|y|⁺ - uᵀ|y|⁻ + lᵥᵀ|z|⁺ - uᵥᵀ|z|⁻ + c0
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
    # the objective constant cancels out in the gap, but not in its scale
    # both objectives are those of the original problem, hence the division by `β γ` (the
    # quadratic term `x̃ᵀQ̃x̃` of the rescaled problem is `β γ xᵀQx` as well)
    pc_sum = subtract_half!!(pc_sum, xQx)
    dobj = broadcast!!(
        (p, v, b, g, k) -> (p + v) / (b * g) + k, scratch.b1, pc_sum, pv_sum, β, γ, c0
    )
    cx = colsum!!(scratch.b2, @. scratch.x = c * x)
    cx = add_half!!(cx, xQx)
    pobj = broadcast!!((cx, b, g, k) -> cx / (b * g) + k, scratch.b2, cx, β, γ, c0)

    err.gap = broadcast!!((a, b) -> abs(a - b), err.gap, pobj, dobj)
    err.gap_scale = broadcast!!((a, b) -> one(T) + abs(a) + abs(b), err.gap_scale, dobj, pobj)
    return err
end

"""
    bound_multipliers!(dest, c_At_y, lv, uv, z)

Return the multipliers of the variable bounds for [`kkt_errors!`](@ref): `z` itself when given,
otherwise the projection of the reduced costs `c_At_y`, written into `dest`.
"""
bound_multipliers!(dest, c_At_y, lv, uv, ::Nothing) = @. dest = proj_multiplier(c_At_y, lv, uv)
bound_multipliers!(_, _, _, _, z::AbstractVecOrMat) = z

function kkt_errors!(
        err::KKTErrors,
        scratch::Scratch,
        sol::PrimalDualSolution,
        z_given::Union{Nothing, AbstractVecOrMat},
        milp::MILP{T},
        ::Val{:pdqp},
        ::TerminationNorm,
    ) where {T}
    (; x, y) = sol
    (; c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale) = milp
    # the rescaled problem has bounds `β b`, objective `γ c` and solution `(β x, γ y)`: primal
    # quantities are divided by `β`, dual ones by `γ` and objectives by `β γ`
    β, γ = bound_scale, obj_scale
    # every norm is of a vector in the original space, so the scaled ones are unscaled first;
    # `colmaxabs!!` overwrites its argument, which is why some products are computed twice

    # primal residual and scale
    err.primal_scale = colmaxabs!!(err.primal_scale, @. scratch.y = inv(D1.diag) * combine(lc, uc))
    A_x = mul!(scratch.y, A, x)
    err.primal = colmaxabs!!(err.primal, @. scratch.y = inv(D1.diag) * (A_x - clamp(A_x, lc, uc)))
    A_x = mul!(scratch.y, A, x)
    Ax_norm = colmaxabs!!(scratch.b1, @. scratch.y = inv(D1.diag) * A_x)
    err.primal_scale = batched_apply!!(
        (b, a, β) -> one(T) + max(b, a) / β, err.primal_scale, err.primal_scale, Ax_norm, β
    )
    err.primal = batched_apply!!(/, err.primal, err.primal, β)

    # dual scale
    At_y = mul!(scratch.x, At, y)
    At_y_norm = colmaxabs!!(scratch.b1, @. scratch.x = inv(D2.diag) * At_y)
    Qx = mul_quadratic!(scratch.z, Q, x)
    Qx_norm = colmaxabs!!(scratch.b2, @. scratch.x = inv(D2.diag) * Qx)
    err.dual_scale = colmaxabs!!(err.dual_scale, @. scratch.x = inv(D2.diag) * c)
    err.dual_scale = batched_apply!!(
        (s, a, q, γ) -> one(T) + max(s, a, q) / γ, err.dual_scale, err.dual_scale, At_y_norm, Qx_norm, γ
    )

    # `err.gap` holds `xᵀQx` until the gap itself is computed
    xQx = colsum!!(err.gap, @. scratch.x = x * scratch.z)
    # dual residual, with the reduced costs `c + Qx - Aᵀy` and their projection `z`
    c_At_y = mul!(scratch.x, At, y, -1, false)
    @. c_At_y += c + scratch.z
    z = bound_multipliers!(scratch.z, c_At_y, lv, uv, z_given)
    # only `z` is needed for the dual objective below, so the residual overwrites the reduced costs
    @. c_At_y = inv(D2.diag) * (c_At_y - z)
    err.dual = colmaxabs!!(err.dual, c_At_y)
    err.dual = batched_apply!!(/, err.dual, err.dual, γ)

    # objectives, as in the default definition
    pc = @. scratch.y = (
        safeprod_left(lc, positive_part(y)) - safeprod_left(uc, negative_part(y))
    )
    pv = @. scratch.x = (
        safeprod_left(lv, positive_part(z)) - safeprod_left(uv, negative_part(z))
    )
    pc_sum = colsum!!(scratch.b1, pc)
    pv_sum = colsum!!(scratch.b2, pv)
    dobj = batched_apply!!(
        (p, v, b, g, k) -> (p + v) / (b * g) + k, scratch.b1, pc_sum, pv_sum, β, γ, c0
    )
    # once only: for a batch, the scaling happens in place
    xQx = scaled_quadratic_term(xQx, β, γ)
    dobj = subtract_half!!(dobj, xQx)
    cx = colsum!!(scratch.b2, @. scratch.x = c * x)
    pobj = batched_apply!!((cx, b, g, k) -> cx / (b * g) + k, scratch.b2, cx, β, γ, c0)
    pobj = add_half!!(pobj, xQx)

    err.gap = broadcast!!((a, b) -> abs(a - b), err.gap, pobj, dobj)
    err.gap_scale = broadcast!!((a, b) -> one(T) + max(abs(a), abs(b)), err.gap_scale, dobj, pobj)
    return err
end
