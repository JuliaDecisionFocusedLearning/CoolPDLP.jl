#=
Infeasibility detection

When an LP is primal or dual infeasible, the PDHG iterates diverge, and both the normalized
iterates and the differences between consecutive iterates converge to a direction whose primal
or dual part is an infeasibility certificate (Applegate et al., "Infeasibility detection with
primal-dual hybrid gradient for large-scale linear programming", and the PDLP paper,
arXiv:2106.04756). At every termination check, a few candidate directions are projected onto
the sign conditions of a ray, then tested on both the original (unscaled) and the preconditioned problem:

- a primal ray `d` certifies dual infeasibility when `A * d` lies in the recession cone of
  `[lc, uc]`, `d` in that of `[lv, uv]` (enforced by the projection), `cᵀd < 0` and, for a
  quadratic program, `Q * d = 0` (the objective must not curve up along the ray; `‖Q d‖` is
  added to the violation);
- a dual ray `y` certifies primal infeasibility when its reduced cost `r = -Aᵀy` pairs with
  finite variable bounds only (`r > 0` needs a finite lower bound and `r < 0` a finite upper
  bound, while the projection makes `y` pair with finite constraint bounds), and its dual
  objective `lcᵀ|y|⁺ - ucᵀ|y|⁻ + lvᵀ|r|⁺ - uvᵀ|r|⁻` is positive.

The error of a ray is its violation of the cone conditions divided by its
objective, which does not depend on the scale of the ray. A ray only qualifies if its objective
exceeds a small fraction of the sum of its absolute terms, which rules out rounding noise.

The objectives are the same in the scaled and original spaces, since `D1` and `D2` cancel out.
=#

"""
    InfeasibilityWorkspace

Buffers for [`infeasibility_check!`](@ref).

# Fields

$(TYPEDFIELDS)
"""
mutable struct InfeasibilityWorkspace{T <: Number, V <: AbstractVecOrMat{T}, B <: BatchedNumber}
    "best rays of the last check, in the scaled space: `ray.x` is a primal ray and `ray.y` a dual ray"
    const ray::PrimalDualSolution{T, V}
    "candidate ray under evaluation"
    const cand::PrimalDualSolution{T, V}
    "per-column scratch (error of the candidate)"
    e::B
    "per-column scratch"
    t1::B
    "per-column scratch"
    t2::B
    "per-column scratch"
    t3::B
    "per-column scratch"
    t4::B
end

"""
    InfeasibilityWorkspace(sol, infeasibility_tol)

Allocate the buffers of the infeasibility detection, or return `nothing` when it is disabled
(`infeasibility_tol === nothing`), so that a disabled detection allocates nothing.
"""
function InfeasibilityWorkspace(sol::PrimalDualSolution{T}, ::Number) where {T}
    z() = batched_expand(sol.x, zero(T))
    return InfeasibilityWorkspace(zero(sol), zero(sol), z(), z(), z(), z(), z())
end
InfeasibilityWorkspace(::PrimalDualSolution, ::Nothing) = nothing

"""
    ray_buffer(sol, infeasibility_tol)

Return a zero solution shaped like `sol` for the infeasibility detection, or `nothing` when it
is disabled.
"""
ray_buffer(sol::PrimalDualSolution, ::Number) = zero(sol)
ray_buffer(::PrimalDualSolution, ::Nothing) = nothing

instance(::Nothing, ::Int) = nothing

function instance(ws::InfeasibilityWorkspace, i::Int)
    return InfeasibilityWorkspace(
        instance(ws.ray, i), instance(ws.cand, i),
        instance_num(ws.e, i), instance_num(ws.t1, i), instance_num(ws.t2, i),
        instance_num(ws.t3, i), instance_num(ws.t4, i),
    )
end

"Project `d` onto the directions along which a variable with bounds `l` and `u` is unbounded."
@inline function project_primal_ray(d, l, u)
    d1 = ifelse(isfinite(l), max(d, zero(d)), d)
    return ifelse(isfinite(u), min(d1, zero(d1)), d1)
end

"Project `y` onto the multipliers that pair with finite bounds `l` (for `y > 0`) and `u` (for `y < 0`)."
@inline function project_dual_ray(y, l, u)
    y1 = ifelse(isfinite(l), y, min(y, zero(y)))
    return ifelse(isfinite(u), y1, max(y1, zero(y1)))
end

"How much `ad = (A * d)ᵢ` leaves the recession cone of `[l, u]`."
@inline function primal_ray_violation(ad, l, u)
    return ifelse(isfinite(l), negative_part(ad), zero(ad)) +
        ifelse(isfinite(u), positive_part(ad), zero(ad))
end

"How much the reduced cost `r` pairs with infinite bounds `l` (for `r > 0`) or `u` (for `r < 0`)."
@inline function dual_ray_violation(r, l, u)
    return ifelse(isfinite(l), zero(r), positive_part(r)) +
        ifelse(isfinite(u), zero(r), negative_part(r))
end

"Dual objective term `l⋅|v|⁺ - u⋅|v|⁻` of a multiplier `v`, ignoring infinite bounds."
@inline bound_term(v, l, u) = safeprod_left(l, positive_part(v)) - safeprod_left(u, negative_part(v))

"Absolute value of each summand of [`bound_term`](@ref), added up."
@inline bound_term_abs(v, l, u) = abs(safeprod_left(l, positive_part(v))) + abs(safeprod_left(u, negative_part(v)))

"""
    ray_error(violation, objective, objective_abs, tol)

Certificate error of a ray: `violation / objective` if `objective` is positive and at least
`tol * objective_abs`, `Inf` otherwise.
"""
@inline function ray_error(violation, objective, objective_abs, tol)
    qualifies = (objective > zero(objective)) & (objective >= tol * objective_abs)
    return ifelse(qualifies, violation / objective, oftype(violation, Inf))
end

"""
    form_candidate!(cand, to, from, milp, active)

Fill `cand` with the projected difference `to - from` (or `to` itself if `from` is `nothing`).
"""
function form_candidate!(
        cand::PrimalDualSolution, to::PrimalDualSolution, from::PrimalDualSolution, milp::MILP,
        active,
    )
    (; lv, uv, lc, uc) = milp
    masked_broadcast!((t, f, l, u) -> project_primal_ray(t - f, l, u), cand.x, active, to.x, from.x, lv, uv)
    masked_broadcast!((t, f, l, u) -> project_dual_ray(t - f, l, u), cand.y, active, to.y, from.y, lc, uc)
    return cand
end

function form_candidate!(
        cand::PrimalDualSolution, to::PrimalDualSolution, ::Nothing, milp::MILP, active,
    )
    (; lv, uv, lc, uc) = milp
    masked_broadcast!(project_primal_ray, cand.x, active, to.x, lv, uv)
    masked_broadcast!(project_dual_ray, cand.y, active, to.y, lc, uc)
    return cand
end

"""
    primal_ray_error!!(dest, ws, scratch, d, milp, tol)

Compute the certificate error of the primal ray `d` (scaled space), column by column, into `dest`.
The violation is measured both on the original problem and on the preconditioned one, and the
error is the larger of the two (see [`dual_ray_error!!`](@ref)).

Costs one multiplication by `A`, and one by `Q` for a quadratic program (both counted in the
pass that [`infeasibility_check!`](@ref) charges to `stats.ray_passes`).
"""
function primal_ray_error!!(
        dest::BatchedNumber, ws::InfeasibilityWorkspace, scratch::Scratch,
        d::AbstractVecOrMat, milp::MILP, tol::Number,
    )
    (; c, A, Q, lc, uc, D1, D2, bound_scale, obj_scale) = milp
    Ad = mul!(scratch.y, A, d)
    viol = @. scratch.y = primal_ray_violation(Ad, lc, uc)
    # each term is brought back to the original problem (see `kkt_errors!`): for a ray `d` of
    # the rescaled problem, `d / β` is one of the original problem, `A d` scales like `β`,
    # `Q d` like `γ` and `cᵀd` like `β γ`; the preconditioned violation (in `dest`) only
    # leaves out the diagonal scalings `D1` and `D2`
    dest = colnorm!!(dest, viol)
    dest = broadcast!!(/, dest, dest, bound_scale)
    ws.t1 = colnorm!!(ws.t1, @. scratch.y = inv(D1.diag) * viol)
    ws.t1 = broadcast!!(/, ws.t1, ws.t1, bound_scale)
    # ‖D2⁻¹ Q d‖ is the violation of `Q d = 0` in the original space (`Q d = D2⁻¹ Q̃ d̃`). It is
    # spelled out rather than delegated to `quadratic_norm!!`: under DispatchDoctor, each extra
    # layer of `@stable` calls makes its compile-time stability check likelier to stop folding,
    # and then it runs (and allocates) at every check. `isquadratic` is a constant for a
    # `ZeroMatrix`, so a linear program skips the block at compile time.
    if isquadratic(Q)
        Qd = quadratic_mul!(scratch.z, Q, d)
        ws.t2 = colnorm!!(ws.t2, Qd)
        dest = broadcast!!((v, q, g) -> v + q / g, dest, dest, ws.t2, obj_scale)
        ws.t4 = colnorm!!(ws.t4, @. scratch.z = inv(D2.diag) * Qd)
        ws.t1 = broadcast!!((v, q, g) -> v + q / g, ws.t1, ws.t1, ws.t4, obj_scale)
    end
    ws.t1 = broadcast!!(max, ws.t1, ws.t1, dest)
    ws.t2 = colsum!!(ws.t2, @. scratch.x = c * d)
    ws.t3 = colsum!!(ws.t3, @. scratch.x = abs(c * d))
    return broadcast!!(dest, ws.t1, ws.t2, ws.t3, bound_scale, obj_scale) do v, o, oa, b, g
        ray_error(v, -o / (b * g), oa / (b * g), tol)
    end
end

"""
    dual_ray_error!!(dest, ws, scratch, y, milp, tol)

Compute the certificate error of the dual ray `y` (scaled space), column by column, into `dest`.

The violation is measured both on the original problem and on the preconditioned one (without
`D1` and `D2`, but with the scalar rescalings undone), and the error is the larger of the two:
a near-certificate of a badly scaled feasible LP can look exact on the original problem only.

Costs one multiplication by `Aᵀ`.
"""
function dual_ray_error!!(
        dest::BatchedNumber, ws::InfeasibilityWorkspace, scratch::Scratch,
        y::AbstractVecOrMat, milp::MILP, tol::Number,
    )
    (; lv, uv, At, lc, uc, D2, bound_scale, obj_scale) = milp
    # literals rather than `-one(T)` and `zero(T)`, which are traced under Reactant
    r = mul!(scratch.x, At, y, -1, false)
    viol = @. scratch.z = dual_ray_violation(r, lv, uv)
    ws.t4 = colnorm!!(ws.t4, viol)
    ws.t1 = colnorm!!(ws.t1, @. scratch.z = inv(D2.diag) * viol)
    ws.t1 = broadcast!!(max, ws.t1, ws.t1, ws.t4)
    ws.t2 = colsum!!(ws.t2, @. scratch.y = bound_term(y, lc, uc))
    ws.t3 = colsum!!(ws.t3, @. scratch.z = bound_term(r, lv, uv))
    ws.t2 = broadcast!!(+, ws.t2, ws.t2, ws.t3)
    ws.t3 = colsum!!(ws.t3, @. scratch.y = bound_term_abs(y, lc, uc))
    ws.t4 = colsum!!(ws.t4, @. scratch.z = bound_term_abs(r, lv, uv))
    ws.t3 = broadcast!!(+, ws.t3, ws.t3, ws.t4)
    # back to the original problem (see `kkt_errors!`): `Aᵀy` scales like `γ` and the dual
    # objective like `β γ`
    return broadcast!!(dest, ws.t1, ws.t2, ws.t3, bound_scale, obj_scale) do v, o, oa, b, g
        ray_error(v / g, o / (b * g), oa / (b * g), tol)
    end
end

"""
    keep_better!!(best_ray, best_err, cand, cand_err, active)

Replace the columns of `best_ray` by those of `cand` where `cand_err < best_err`, and return the
column-wise minimum of the errors.
"""
function keep_better!!(
        best_ray::AbstractVecOrMat, best_err::BatchedNumber,
        cand::AbstractVecOrMat, cand_err::BatchedNumber, active,
    )
    masked_broadcast!(best_ray, active, transpose(cand_err), transpose(best_err), cand, best_ray) do e, b, c, r
        ifelse(e < b, c, r)
    end
    return masked_apply!!(min, best_err, active, best_err, cand_err)
end

"""
    infeasibility_check!(stats, infeas, scratch, candidate1, candidate2, milp, tol, active)

Look for infeasibility certificates among two candidate rays, each given as a `(to, from)` pair
(see [`ray_candidates`](@ref)), and store the errors of the best ones in
`stats.primal_ray_error` and `stats.dual_ray_error`, and the rays themselves in `infeas.ray`.

Each candidate costs one multiplication by `A` and one by `Aᵀ`, plus one by `Q` for a quadratic
program, i.e. one KKT pass with the same convention as an iteration (which also multiplies by
`A`, `Aᵀ` and `Q` once each). These two passes go to `stats.ray_passes`, not `stats.kkt_passes`:
the detection does not eat into `max_kkt_passes`. Does nothing at all when the detection is
disabled (`infeas === nothing`).
"""
function infeasibility_check!(
        stats::ConvergenceStats, infeas::InfeasibilityWorkspace, scratch::Scratch,
        (to1, from1)::Tuple, (to2, from2)::Tuple, milp::MILP, tol::Number, active,
    )
    (; ray, cand) = infeas
    # the rays of a frozen instance (see `active_mask`) stay those it terminated with; their
    # errors are recomputed for every column below, which leaves a frozen instance's errors
    # unchanged only because its ray is masked and the computation is deterministic

    form_candidate!(ray, to1, from1, milp, active)
    stats.primal_ray_error = primal_ray_error!!(
        stats.primal_ray_error, infeas, scratch, ray.x, milp, tol
    )
    stats.dual_ray_error = dual_ray_error!!(
        stats.dual_ray_error, infeas, scratch, ray.y, milp, tol
    )

    form_candidate!(cand, to2, from2, milp, true)
    infeas.e = primal_ray_error!!(infeas.e, infeas, scratch, cand.x, milp, tol)
    stats.primal_ray_error = keep_better!!(ray.x, stats.primal_ray_error, cand.x, infeas.e, active)
    infeas.e = dual_ray_error!!(infeas.e, infeas, scratch, cand.y, milp, tol)
    stats.dual_ray_error = keep_better!!(ray.y, stats.dual_ray_error, cand.y, infeas.e, active)

    stats.ray_passes += 2
    return nothing
end

function infeasibility_check!(
        ::ConvergenceStats, ::Nothing, ::Scratch, ::Tuple, ::Tuple, ::MILP, ::Nothing, ::Any,
    )
    return nothing
end

"""
    export_certificate!(stats, infeas, milp)

Copy the rays of `infeas` into `stats.ray`, back in the original (unscaled) space. Does nothing
when the detection is disabled.
"""
function export_certificate!(stats::ConvergenceStats, infeas::InfeasibilityWorkspace, milp::MILP)
    (; D1, D2) = milp
    # rays are directions: the bound and objective rescaling only changes their length
    @. stats.ray.x = D2.diag * infeas.ray.x
    @. stats.ray.y = D1.diag * infeas.ray.y
    return nothing
end

export_certificate!(::ConvergenceStats, ::Nothing, ::MILP) = nothing
