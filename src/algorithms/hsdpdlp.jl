#=
H-PDLP: restarted reflected Halpern PDHG on a box-cone homogeneous self-dual embedding
(Lu & Tang, arXiv:2610.17695).

The LP `min cᵀx s.t. lc ≤ Ax ≤ uc, lv ≤ x ≤ uv` is embedded as `z ∈ K, Qz ∈ K*` with
`z = (τ, x, α, y)`, `K = K[lv, uv] × K[lc, uc]*`, where `K[l, u] = cl{(τ, x): τ > 0, lτ ≤ x ≤ uτ}`
is a box cone, and the skew-symmetric `Qz = (-cᵀx - α, cτ - Aᵀy, τ, Ax)`. A solution with `τ > 0`
gives the optimal pair `(x, y) / τ`, one with `τ = 0` and `κ = -cᵀx - p(-y; lc, uc) - p(-r; lv, uv)`
positive an infeasibility certificate. The iteration therefore always has something to converge
to, whatever the status of the LP.

The PDHG operator `T` of the embedding (eq. 28 of the paper) projects the dual block `(α, y)` onto
`K[lc, uc]*` (through Moreau's identity, a Euclidean projection onto `K[lc, uc]`) and the primal
block `(τ, x)` onto `K[lv, uv]` in the metric `diag(1/μ, I/η)`, implicitly in `τ`. Both reduce to a
scalar root, found by the safeguarded Newton method of Algorithm 1.
=#

"""
    HSDPDLP(args...; kwargs...)

Shortcut for [`Algorithm{:HSDPDLP}`](@ref): H-PDLP (Lu & Tang, arXiv:2610.17695), the restarted
reflected Halpern PDHG of [`HPDLP`](@ref) applied to a box-cone homogeneous self-dual embedding of
the LP. Linear programs only.

Optimality and infeasibility are both read off the iterates of the embedding: an instance stops as
`MOI.OPTIMAL` when `(x, y) / τ` meets `termination_reltol`, as `MOI.INFEASIBLE` or
`MOI.DUAL_INFEASIBLE` when `y` or `x` (or their displacement since the last restart) is a
certificate within `infeasibility_tol`, exposed in `stats.ray` like the ray detection of the other
algorithms.

Its defaults follow the paper: step sizes `η = λ/ω`, `σ = λω` for `x` and `(α, y)` and `μ = ν/(λω)`
for `τ`, with `λ = ν / ‖A‖₂` and `ν = invnorm_scaling = 0.998`; the restarts of [`HPDLP`](@ref)
checked every `200` iterations; a PI primal weight controller (`pid_kp = 0.97`, `pid_ki = 0.05`,
`pid_integral_discount = 0.3`) held in the certificate regime.

Deviations from the paper: the Newton tolerance is `1e-9` (see `HSD_NEWTON_TOL`); the primal
weight stays within `[1e-4, 1e4]` with anti-windup; and the preconditioner is the package default
(no geometric-mean passes). The optimality test tries both reduced-cost candidates of the paper
(see [`refine_kkt_errors!`](@ref)). Infeasibility can only be reported when
`infeasibility_tol` is not `nothing`.
"""
function HSDPDLP(args...; kwargs...)
    return Algorithm{:HSDPDLP}(
        args...;
        invnorm_scaling = 0.998,
        necessary_decay = 0.5,
        check_every = 200,
        pid_kp = 0.97,
        pid_ki = 0.05,
        pid_kd = 0.0,
        pid_integral_discount = 0.3,
        kwargs...
    )
end

"relative tolerance `ϵ_N` of the safeguarded Newton method (the paper's `1e-6` stalls the outer iteration)"
const HSD_NEWTON_TOL = 1.0e-9
"factor around the initial primal weight `1` which the weight is kept within (a safeguard, not in the paper)"
const HSD_OMEGA_WINDOW = 1.0e4
"maximum number of iterations of the safeguarded Newton method, as in the paper"
const HSD_NEWTON_MAXITER = 100
"number of Newton iterations between two convergence tests of [`boxcone_root!`](@ref), each of which reduces over the batch"
const HSD_NEWTON_CHECK_EVERY = 4

"""
    HSDPoint

Point `(τ, x, α, y)` of the homogeneous self-dual embedding, with one `τ` and one `α` per instance.

# Fields

$(TYPEDFIELDS)
"""
mutable struct HSDPoint{X <: PrimalDualSolution, S <: BatchedNumber}
    "the `(x, y)` block"
    const xy::X
    "homogeneous scale `τ`"
    τ::S
    "dual scalar `α`, which bounds the support function of the constraint box"
    α::S
end

function Base.copy(p::HSDPoint)
    return HSDPoint(copy(p.xy), batched_apply!!(identity, batched_similar(p.τ), p.τ), batched_apply!!(identity, batched_similar(p.α), p.α))
end

function instance(p::HSDPoint, i::Int)
    return HSDPoint(instance(p.xy, i), instance_num(p.τ, i), instance_num(p.α, i))
end

"""
    HSDWorkspace

Buffers of the HSD operator and of its scalar Newton solves.

# Fields

$(TYPEDFIELDS)
"""
mutable struct HSDWorkspace{V <: AbstractVecOrMat, S <: BatchedNumber, B <: BatchedNumber}
    "primal shift `x̃` of the implicit primal projection"
    const xt::V
    "point `-ỹ` projected onto the constraint box cone"
    const wt::V
    "primal buffer (length `nvar`)"
    const bx::V
    "dual buffer (length `ncons`)"
    const by::V
    "scalar part of the point projected (`-α̃` for the dual block, `τ̃` for the primal block)"
    tt::S
    "residual `g` of the scalar root"
    g::S
    "Newton denominator `d` of the scalar root"
    d::S
    "current root estimate"
    t::S
    "lower end of the bracket"
    lo::S
    "upper end of the bracket"
    hi::S
    "upper end of the initial bracket, the scale below which a root counts as zero"
    hi0::S
    "length of the last Newton step"
    Δ::S
    "whether the scalar root of each instance has converged"
    done::B
    "next root estimate"
    t2::S
    "next lower end"
    lo2::S
    "next upper end"
    hi2::S
    "next step length"
    Δ2::S
    "next convergence flags"
    done2::B
    "warm start of the dual projection: its root at the previous step"
    warm::S
    "step size `σ = λω` of `(α, y)`"
    σ::S
    "step size `η = λ/ω` of `x`"
    ηx::S
    "step size `μ = ν/(λω)` of `τ`"
    μ::S
    "lower bound `1/μ` on the slope of the primal root"
    dmin::S
    "total number of Newton iterations so far (both roots), a diagnostic"
    newton_iters::S
    "inverse `1/η` of the step of `x`, for the primal root (Reactant cannot divide by it inside a broadcast)"
    inv_ηx::S
    "factor `1 / (τ γb)` recovering `x` from the PDHG point"
    scale_x::S
    "factor `1 / (τ γc)` recovering `y` from the PDHG point"
    scale_y::S
    "cross term `⟨AΔx, Δy⟩` of the fixed-point residual"
    cross::S
    "primal part `Σ bound_term(r) - cᵀx` of `κ` (eq. 49)"
    κ_primal::S
    "dual part `Σ bound_term(y)` of `κ` (eq. 49)"
    κ_dual::S
    "certificate-regime indicator `ρ = κ⁺ / (τ + κ⁺)` (eq. 50)"
    ρ::S
    "primal movement `δp` since the anchor (eq. 46)"
    δp::S
    "dual movement `δd` since the anchor (eq. 46)"
    δd::S
    "homogeneous rescaling factor of the restart point (eq. 44)"
    restart_scale::S
    "scale `1 / (1 + ‖(lc, uc)‖)` of the bounds in the embedding"
    γb::S
    "scale `1 / (1 + ‖c‖)` of the objective in the embedding"
    γc::S
end

function HSDWorkspace(sol::PrimalDualSolution{T}, γb::BatchedNumber, γc::BatchedNumber) where {T}
    num(v) = batched_expand(sol.x, convert(T, v))
    flag() = batched_expand(sol.x, false)
    return HSDWorkspace(
        similar(sol.x), similar(sol.y), similar(sol.x), similar(sol.y),
        num(0), num(0), num(1), num(0), num(0), num(0), num(0), num(Inf), flag(),
        num(0), num(0), num(0), num(Inf), flag(),
        num(0), num(1), num(1), num(1), num(1), num(0), num(0), num(0), num(0), num(0), num(0), num(0), num(0), num(0), num(0), num(0),
        γb, γc,
    )
end

function instance(ws::HSDWorkspace, i::Int)
    n(v) = instance_num(v, i)
    return HSDWorkspace(
        instance_vec(ws.xt, i), instance_vec(ws.wt, i), instance_vec(ws.bx, i), instance_vec(ws.by, i),
        n(ws.tt), n(ws.g), n(ws.d), n(ws.t), n(ws.lo), n(ws.hi), n(ws.hi0), n(ws.Δ), n(ws.done),
        n(ws.t2), n(ws.lo2), n(ws.hi2), n(ws.Δ2), n(ws.done2),
        n(ws.warm), n(ws.σ), n(ws.ηx), n(ws.μ), n(ws.dmin), n(ws.newton_iters),
        n(ws.inv_ηx), n(ws.scale_x), n(ws.scale_y), n(ws.cross), n(ws.κ_primal), n(ws.κ_dual),
        n(ws.ρ), n(ws.δp), n(ws.δd), n(ws.restart_scale),
        n(ws.γb), n(ws.γc),
    )
end

"""
    HSDPDLPState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct HSDPDLPState{
        T <: Number,
        V <: AbstractVecOrMat{T},
        P <: HSDPoint,
        M <: NamedTuple,
        W <: HSDWorkspace,
        SS <: StepSizes,
        Sc <: Scratch,
        HS <: HalpernStats,
        CS <: ConvergenceStats,
        IW <: Union{Nothing, InfeasibilityWorkspace},
        E <: KKTErrors,
    } <: AbstractState{T, V}
    "recovered solution `(x, y) / τ` of the PDHG point `T(z)` at the last check, which is the reported solution"
    sol::PrimalDualSolution{T, V}
    "PDHG point `T(z)` of the Halpern iterate before the last step"
    Tz::P
    "current Halpern iterate `z`"
    z::P
    "Halpern anchor, set at the last restart"
    z_anchor::P
    "the LP with its objective and bounds normalized, which the embedding is built on"
    hsd::M
    "operator buffers"
    ws::W
    "step sizes: `η` holds the base step `λ` and `ω` the primal weight"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "Halpern and restart bookkeeping"
    halpern::HS
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
    "active-set reduced costs of the second optimality candidate (see [`refine_kkt_errors!`](@ref))"
    r_act::V
    "KKT errors of the second optimality candidate"
    err_act::E
end

nbinstances((; sol)::HSDPDLPState) = nbinstances(sol)
function instance(state::HSDPDLPState, i::Int)
    return HSDPDLPState(
        instance(state.sol, i),
        instance(state.Tz, i),
        instance(state.z, i),
        instance(state.z_anchor, i),
        hsd_instance(state.hsd, i),
        instance(state.ws, i),
        instance(state.step_sizes, i),
        instance(state.scratch, i),
        instance(state.halpern, i),
        instance(state.stats, i),
        instance(state.infeas, i),
        instance_vec(state.r_act, i),
        instance(state.err_act, i),
    )
end

"The `i`-th instance of the scaled data of [`hsd_problem`](@ref)."
function hsd_instance(hsd::NamedTuple, i::Int)
    (; c, lv, uv, A, At, lc, uc) = hsd
    v(x) = instance_vec(x, i)
    return (; c = v(c), lv = v(lv), uv = v(uv), A, At, lc = v(lc), uc = v(uc))
end

"""
    hsd_bound_norm(l, u)

Contribution of one pair of bounds to `‖(l, u)‖₂` as defined in the paper: both finite ends of a
proper interval, an equality once, infinite ends never.
"""
@inline function hsd_bound_norm(l, u)
    us = ifelse(isfinite(u), u^2, zero(u))
    ls = ifelse(isfinite(l) & (l != u), l^2, zero(l))
    return sqrt(us + ls)
end

"""
    hsd_problem(milp, sol)

Return the data of the LP that the embedding is built on, as a named tuple `(c, lv, uv, A, At, lc, uc)`, with its bounds scaled by `γb = 1 / (1 + ‖(lc, uc)‖₂)`
and its objective by `γc = 1 / (1 + ‖c‖₂)` (the last stage of the preconditioning of the paper),
together with `γb` and `γc`, one per instance.

A solution `(x, y)` of the scaled LP is `(γb x, γc y)` for the original one, and certificates are
the same in both.
"""
function hsd_problem(milp::MILP{T}, sol::PrimalDualSolution{T}) where {T}
    (; c, lv, uv, A, At, lc, uc) = milp
    nb = broadcast(n -> inv(one(T) + n), colnorm(hsd_bound_norm.(lc, uc)))
    nc = broadcast(n -> inv(one(T) + n), colnorm(c))
    γb = batched_expand(sol.x, nb)
    γc = batched_expand(sol.x, nc)
    γbr, γcr = transpose(γb), transpose(γc)
    # a named tuple rather than a `MILP`, whose inner constructor Reactant cannot rebuild a traced
    # state with; `A` and `At` are those of `milp`, shared
    hsd = (; c = c .* γcr, lv = lv .* γbr, uv = uv .* γbr, A, At, lc = lc .* γbr, uc = uc .* γbr)
    return hsd, γb, γc
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:HSDPDLP};
        starting_time::Float64
    ) where {T, V}
    hsd, γb, γc = hsd_problem(milp, sol)
    # the practical anchor (1, x⁰, 1, y⁰) of the paper, (1, 0, 1, 0) from the default start
    xy = PrimalDualSolution(sol.x .* transpose(γb), sol.y .* transpose(γc))
    z = HSDPoint(xy, batched_expand(sol.x, one(T)), batched_expand(sol.x, one(T)))
    Tz = copy(z)
    z_anchor = copy(z)
    ω = batched_expand(sol.x, one(T))
    λ = batched_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    step_sizes = StepSizes(; η = λ, ω)
    ws = HSDWorkspace(sol, γb, γc)
    scratch = Scratch(sol)
    halpern = HalpernStats(sol, ω)
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp, algo.generic.termination_norm)
    tol = algo.termination.infeasibility_tol
    stats = ConvergenceStats(err; ray = ray_buffer(sol, tol), starting_time, extra_products = 2)
    infeas = InfeasibilityWorkspace(sol, tol)
    return HSDPDLPState(;
        sol = copy(sol), Tz, z, z_anchor, hsd, ws, step_sizes, scratch, halpern, stats, infeas,
        r_act = zero(sol.x), err_act = KKTErrors(sol),
    )
end

"""
    solve!(state::HSDPDLPState, milp, algo)

Run H-PDLP until termination, with the loop structure of [`HPDLP`](@ref).
"""
function solve!(
        state::HSDPDLPState,
        milp::MILP,
        algo::Algorithm{:HSDPDLP}
    )
    check_every = algo.generic.check_every
    check_every >= 2 || throw(ArgumentError("HSDPDLP needs `check_every >= 2`, got $check_every"))
    prog = init_progress("HSDPDLP iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        must_terminate = check_pass!(state, milp, algo, prog)
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    check_pass!(state::HSDPDLPState, milp, algo, prog = NoProgress())

Run one block of `check_every` steps, recover the solution, check termination and restart;
return whether to stop (see [`solve!`](@ref)).
"""
function check_pass!(
        state::HSDPDLPState,
        milp::MILP,
        algo::Algorithm{:HSDPDLP},
        prog = NoProgress()
    )
    hsd_step!(state, algo, Val(true))
    seed_fixed_point_error!(state.halpern)
    next_progress!(prog, state)
    @trace for _ in 1:(algo.generic.check_every - 2)
        hsd_step!(state, algo, Val(false))
        next_progress!(prog, state)
    end
    hsd_step!(state, algo, Val(true))
    next_progress!(prog, state)
    recover_solution!(state.sol, state.Tz, state.ws, active_mask(state.stats))
    must_terminate = termination_check!(state, milp, algo)
    halpern_restart_check!(state.halpern, algo.restart, must_terminate, active_mask(state.stats))
    hsd_restart!(state, algo)
    state.stats.newton_iterations = masked_apply!!(
        n -> n + zero(n), state.stats.newton_iterations, active_mask(state.stats), state.ws.newton_iters
    )
    return must_terminate
end

"""
    refine_kkt_errors!(state, milp, norm)

Second optimality candidate of the paper (§6.2), tried by `termination_check!` after the
projection `Π_R(c - Aᵀy)` of [`kkt_errors!`](@ref): the active-set reduced cost `r_act`: the reduced cost `q = c - Aᵀy` kept only
where the last primal projection put `x` on a bound and `q` has the sign of a multiplier of that
bound, zero elsewhere. Where `Π_R(q)` charges a nonzero multiplier of an inactive bound to the
gap, this one charges it to the dual residual instead. The errors of the better candidate end up
in `state.stats.err`. Costs three products by `A` or `Aᵀ`.

A frozen instance of a batch keeps its `r_act`, hence its errors.
"""
function refine_kkt_errors!(state::HSDPDLPState, milp::MILP, norm::TerminationNorm)
    (; sol, Tz, hsd, scratch, stats, r_act, err_act) = state
    (; c, At) = milp
    At_y = mul!(scratch.x, At, sol.y)
    τr = transpose(Tz.τ)
    masked_broadcast!(active_reduced_cost, r_act, active_mask(stats), c, At_y, Tz.xy.x, hsd.lv, hsd.uv, τr)
    kkt_errors!(err_act, scratch, sol, r_act, milp, norm)
    stats.extra_products += 3
    keep_smaller_errors!(stats.err, err_act, scratch)
    return nothing
end

"""
Active-set reduced cost of one coordinate: `q = c - a` if `x` sits on a finite bound of the box
cone (where `box_clip` puts it exactly) and `q` has the sign of that bound's multiplier, else `0`.
"""
@inline function active_reduced_cost(c, a, x, l, u, τ)
    q = c - a
    low = isfinite(l) & (x <= l * τ)
    up = isfinite(u) & (x >= u * τ)
    return ifelse((low & (q > zero(q))) | (up & (q < zero(q))), q, zero(q))
end

"Overwrite, instance by instance, the errors in `err` by those in `alt` where `alt` is relatively smaller."
function keep_smaller_errors!(err::KKTErrors, alt::KKTErrors, scratch::Scratch)
    a = relative!!(scratch.b1, err)
    b = relative!!(scratch.b2, alt)
    pick(a, b, x, y) = ifelse(b < a, y, x)
    err.primal = batched_apply!!(pick, err.primal, a, b, err.primal, alt.primal)
    err.primal_scale = batched_apply!!(pick, err.primal_scale, a, b, err.primal_scale, alt.primal_scale)
    err.dual = batched_apply!!(pick, err.dual, a, b, err.dual, alt.dual)
    err.dual_scale = batched_apply!!(pick, err.dual_scale, a, b, err.dual_scale, alt.dual_scale)
    err.gap = batched_apply!!(pick, err.gap, a, b, err.gap, alt.gap)
    err.gap_scale = batched_apply!!(pick, err.gap_scale, a, b, err.gap_scale, alt.gap_scale)
    return err
end

"""
    ray_candidates(state::HSDPDLPState)

For H-PDLP, the candidate certificates are those of the paper: the `(x, y)` block of the PDHG
point `T(z)` itself, and its displacement since the anchor of the last restart (eq. 54).
"""
ray_candidates(state::HSDPDLPState) = ((state.Tz.xy, state.z_anchor.xy), (state.Tz.xy, nothing))

"""
    recover_solution!(sol, Tz, ws, active)

Write the solution `(x, y) / τ` of the original (preconditioned) LP recovered from the PDHG point
`Tz`, undoing the normalization of [`hsd_problem`](@ref). An instance with `τ = 0`, which sits on
the certificate branch, gets zeros.

A frozen instance of a batch (see `active_mask`) keeps the solution of its termination check.
Its `T(z)` no longer moves either (see [`hsd_step!`](@ref)), so the mask only makes sure that
the solution is exactly the one its termination was decided on.
"""
function recover_solution!(sol::PrimalDualSolution, Tz::HSDPoint, ws::HSDWorkspace, active)
    # the scale factors are formed outside the broadcasts: under Reactant, a `Float32` division
    # between traced numbers inside a broadcast fails to compile
    scale(τ, γ) = ifelse(τ > zero(τ), inv(τ * γ), zero(τ))
    ws.scale_x = batched_apply!!(scale, ws.scale_x, Tz.τ, ws.γb)
    ws.scale_y = batched_apply!!(scale, ws.scale_y, Tz.τ, ws.γc)
    sx, sy = transpose(ws.scale_x), transpose(ws.scale_y)
    masked_broadcast!(*, sol.x, active, Tz.xy.x, sx)
    masked_broadcast!(*, sol.y, active, Tz.xy.y, sy)
    return sol
end

"""
    hsd_step!(state, algo, Val(fixed_point_error))

Apply one reflected Halpern step to the iterate of the embedding, keeping `T(z)` in `state.Tz`.
With `Val(true)`, also store the fixed-point residual `‖z - T(z)‖_P` in `state.halpern.fpe`.
"""
function hsd_step!(
        state::HSDPDLPState, algo::Algorithm{:HSDPDLP}, ::Val{fixed_point_error}
    ) where {fixed_point_error}
    (; Tz, z, z_anchor, hsd, ws, step_sizes, scratch, halpern) = state
    γ = algo.step_size.reflection_coefficient
    active = active_mask(state.stats)
    hsd_operator!(Tz, z, ws, hsd, step_sizes, algo.step_size.invnorm_scaling, active)
    if fixed_point_error
        halpern.fpe = hsd_fixed_point_error!!(halpern.fpe, ws, scratch, z, Tz, hsd.A)
        state.stats.extra_products += 1
    end
    w = batched_apply!!(k -> (k + 1) / (k + 2), scratch.b1, halpern.inner)
    # a frozen instance of a batch keeps its `z`, hence its `T(z)` and its solution: its step
    # sizes only change when it restarts, and its Newton roots stay at their warm starts, which
    # are the roots of its last step (see `active_mask` and `boxcone_root!`)
    halpern_combine!(z.xy.x, w, γ, Tz.xy.x, z_anchor.xy.x, active)
    halpern_combine!(z.xy.y, w, γ, Tz.xy.y, z_anchor.xy.y, active)
    z.τ = halpern_combine!!(z.τ, w, γ, Tz.τ, z_anchor.τ, active)
    z.α = halpern_combine!!(z.α, w, γ, Tz.α, z_anchor.α, active)
    halpern.inner = masked_apply!!(k -> k + 1, halpern.inner, active, halpern.inner)
    halpern.total += 1
    state.stats.kkt_passes += 1
    return nothing
end

function halpern_combine!!(
        z::BatchedNumber, w::BatchedNumber, γ::Number, Tz::BatchedNumber, z0::BatchedNumber,
        ::Bool,
    )
    return batched_apply!!(z, w, z, Tz, z0) do w, z, Tz, z0
        w * (γ * (2Tz - z) + (1 - γ) * z) + (1 - w) * z0
    end
end
function halpern_combine!!(
        z::AbstractVector, w::AbstractVector, γ::Number, Tz::AbstractVector, z0::AbstractVector,
        active::AbstractVector,
    )
    return masked_apply!!(z, active, w, z, Tz, z0) do w, z, Tz, z0
        w * (γ * (2Tz - z) + (1 - γ) * z) + (1 - w) * z0
    end
end

"""
    fresh(x)

`x` itself, as a new value: under Reactant, two fields holding the same traced scalar confuse the
tracing of the loops which update them, so an unbatched copy must not just alias its source.
"""
@inline fresh(x::Number) = x + zero(x)
@inline fresh(x::Bool) = x & true

"Copy the per-instance quantity `src` into `dest`, as a [`fresh`](@ref) value when unbatched."
copy_into!!(::Number, src::Number) = fresh(src)
copy_into!!(dest::AbstractVector, src::AbstractVector) = copyto!(dest, src)

"`lτ` for a bound `l` of a box cone, which stays infinite when `l` is"
@inline scaled_bound(l, τ) = ifelse(isfinite(l), l * τ, l)

"`v` clipped into `[lτ, uτ]`, i.e. into the recession cone of `[l, u]` at `τ = 0`"
@inline box_clip(v, l, u, τ) = min(max(v, scaled_bound(l, τ)), scaled_bound(u, τ))

"the bound `v` is clipped to at scale `τ` (`0` if none), for the box-cone root functions"
@inline function clip_bound(v, l, u, τ)
    low = isfinite(l) & (v < scaled_bound(l, τ))
    up = isfinite(u) & (v > scaled_bound(u, τ))
    return ifelse(low, l, ifelse(up, u, zero(l))), low | up
end

"term of `φ'(τ)` (eq. 32) for one coordinate of a Euclidean box-cone projection of `w`"
@inline function euclid_g(w, l, u, τ)
    b, clipped = clip_bound(w, l, u, τ)
    return ifelse(clipped, b * (b * τ - w), zero(w))
end

"term of the slope (eq. 33) for one coordinate of a Euclidean box-cone projection of `w`"
@inline function euclid_d(w, l, u, τ)
    b, clipped = clip_bound(w, l, u, τ)
    return ifelse(clipped, b * b, zero(w))
end

"""
    primal_g(x̃, c, η, η⁻¹, l, u, τ)

Term of `ψ(τ)` (eq. 38) for one coordinate of the implicit primal projection, where
`x(τ) = clip(x̃ - ηcτ, lτ, uτ)`: `-c x(τ)` if it is not clipped, `l (lτ - x̃) / η` if it is clipped
to `lτ` (the terms in `c` cancel out), and likewise for `u`. The inverse step `η⁻¹` is passed in
because, under Reactant, a `Float32` division by a traced number inside a broadcast fails to compile.
"""
@inline function primal_g(xt, c, η, ηi, l, u, τ)
    v = xt - η * c * τ
    b, clipped = clip_bound(v, l, u, τ)
    return ifelse(clipped, b * (b * τ - xt) * ηi, -c * v)
end

"term of the Newton denominator of `ψ` for one coordinate: `ηc²` if free, `b²/η` if clipped to `b`"
@inline function primal_d(xt, c, η, ηi, l, u, τ)
    v = xt - η * c * τ
    b, clipped = clip_bound(v, l, u, τ)
    return ifelse(clipped, b * b * ηi, η * c * c)
end

"""
    dual_residual!!(g, d, buf, wt, tt, lc, uc, t)
    primal_residual!!(g, d, buf, xt, tt, μ, η, η⁻¹, c, lv, uv, t)

Return the residual `g(t)` of the scalar root and its Newton denominator `d(t)`, for the dual
projection (`φ'` of eq. 32 at the point `(tt, wt)`) or the primal one (`ψ` of eq. 38 for the
shift `(tt, xt)`), using `buf` as scratch.
"""
@inline function dual_residual!!(
        g::BatchedNumber, d::BatchedNumber, buf::AbstractVecOrMat, wt::AbstractVecOrMat,
        tt::BatchedNumber, lc::AbstractVecOrMat, uc::AbstractVecOrMat, t::BatchedNumber,
    )
    tr = transpose(t)
    g = colsum!!(g, @. buf = euclid_g(wt, lc, uc, tr))
    g = batched_apply!!((s, t, tt) -> s + t - tt, g, g, t, tt)
    d = colsum!!(d, @. buf = euclid_d(wt, lc, uc, tr))
    d = batched_apply!!(s -> s + one(s), d, d)
    return g, d
end

@inline function primal_residual!!(
        g::BatchedNumber, d::BatchedNumber, buf::AbstractVecOrMat, xt::AbstractVecOrMat,
        tt::BatchedNumber, μ::BatchedNumber, η::BatchedNumber, ηi::BatchedNumber,
        c::AbstractVecOrMat, lv::AbstractVecOrMat, uv::AbstractVecOrMat, t::BatchedNumber,
    )
    tr = transpose(t)
    ηr, ηir = transpose(η), transpose(ηi)
    g = colsum!!(g, @. buf = primal_g(xt, c, ηr, ηir, lv, uv, tr))
    g = batched_apply!!((s, t, tt, μ) -> s + (t - tt) / μ, g, g, t, tt, μ)
    d = colsum!!(d, @. buf = primal_d(xt, c, ηr, ηir, lv, uv, tr))
    d = batched_apply!!((s, μ) -> s + inv(μ), d, d, μ)
    return g, d
end

"""
    newton_update(t, g, d, lo, hi, hi0, Δ, done, dmin, ϵ)

One iteration of Algorithm 1 of the paper for one instance, as a pure function of its state:
return the next `(t, lo, hi, Δ, done)`. A converged instance is left as it is.

Besides the tests of the paper, an instance stops when its next iterate is its current one: the
test `|g| ≤ d_min ϵ/(1+ϵ) t` can be out of reach in floating point when the slope `d` far exceeds
`d_min`, and such a `t` is the root to machine precision.
"""
@inline function newton_update(t, g, d, lo, hi, hi0, Δ, done, dmin, ϵ)
    # a root at zero is only reached through `g(0) ≥ 0`
    converged = (abs(g) <= dmin * ϵ / (1 + ϵ) * t) | (iszero(t) & (g >= zero(g)))
    negative = g < zero(g)
    lo2 = ifelse(negative, t, lo)
    hi2 = ifelse(negative, hi, t)
    mid = (lo2 + hi2) / 2
    # relative to `lo` as in the paper, but with a floor: a root close to zero would otherwise
    # bisect down to the underflow range
    bracketed = hi2 - lo2 <= 2ϵ * max(lo2, ϵ * hi0)
    # with no positive lower end yet, a step below zero goes to zero, where `g(0)` decides
    newton = ifelse(iszero(lo2), max(t - g / d, zero(t)), t - g / d)
    reject = (newton < lo2) | (newton > hi2) | (abs(newton - t) > Δ / 2)
    next = ifelse(reject, mid, newton)
    stalled = next == t
    stop = converged | bracketed | stalled
    t_new = ifelse(done | converged, t, ifelse(bracketed, mid, next))
    Δ_new = ifelse(done | stop, Δ, abs(next - t))
    return t_new, ifelse(done, lo, lo2), ifelse(done, hi, hi2), Δ_new, done | stop
end

"""
    newton_step!(ws, dmin, ϵ)

Apply [`newton_update`](@ref) to every instance, from the residual just evaluated in `ws`.
"""
@inline function newton_step!(ws::HSDWorkspace, dmin::BatchedNumber, ϵ::Number)
    (; t, g, d, lo, hi, hi0, Δ, done) = ws
    # the component is a type parameter, so that each closure infers to a concrete type
    upd(::Val{k}) where {k} = (t, g, d, lo, hi, hi0, Δ, done, dmin) -> newton_update(t, g, d, lo, hi, hi0, Δ, done, dmin, ϵ)[k]
    ws.t2 = batched_apply!!(upd(Val(1)), ws.t2, t, g, d, lo, hi, hi0, Δ, done, dmin)
    ws.lo2 = batched_apply!!(upd(Val(2)), ws.lo2, t, g, d, lo, hi, hi0, Δ, done, dmin)
    ws.hi2 = batched_apply!!(upd(Val(3)), ws.hi2, t, g, d, lo, hi, hi0, Δ, done, dmin)
    ws.Δ2 = batched_apply!!(upd(Val(4)), ws.Δ2, t, g, d, lo, hi, hi0, Δ, done, dmin)
    ws.done2 = batched_apply!!(upd(Val(5)), ws.done2, t, g, d, lo, hi, hi0, Δ, done, dmin)
    ws.t = copy_into!!(ws.t, ws.t2)
    ws.lo = copy_into!!(ws.lo, ws.lo2)
    ws.hi = copy_into!!(ws.hi, ws.hi2)
    ws.Δ = copy_into!!(ws.Δ, ws.Δ2)
    ws.done = copy_into!!(ws.done, ws.done2)
    return nothing
end

"""
    boxcone_root!(ws, (lc, uc) or (c, lv, uv), warm, dmin, active = true)

Find, for every instance, the `τ ≥ 0` with `g(τ) ≥ 0` and `τ g(τ) = 0` by the safeguarded Newton
method of Algorithm 1, warm-started from `warm`, and return it (in `ws.t`).

The iterations are masked per instance and the loop is a `@trace while`, whose trip count is
data-dependent: under Reactant, it compiles to a `while` of the XLA program. It tests for convergence
every `HSD_NEWTON_CHECK_EVERY` iterations, and stops once every instance has converged, or after
`HSD_NEWTON_MAXITER` iterations (rounded up to the next test). The frozen instances of a batch
(`active` false, see `active_mask`) count as converged from the start: they keep their warm
start, and never hold the loop up.
"""
function boxcone_root!(
        ws::HSDWorkspace{V}, data::Union{NTuple{2}, NTuple{3}}, warm::BatchedNumber, dmin::BatchedNumber, active = true,
    ) where {V}
    ϵ = newton_tolerance(eltype(V))
    # start from the warm start itself: its residual sets the bracket, and its Newton step is
    # usually the root already
    ws.t = batched_apply!!(w -> max(w, zero(w)), ws.t, warm)
    newton_residual!(ws, data)
    newton_init!(ws, dmin, active)
    count_newton!(ws)
    newton_step!(ws, dmin, ϵ)
    iter = 1
    @trace while (iter < HSD_NEWTON_MAXITER) & !batched_all(identity, ws.done)
        newton_block!(ws, data, dmin, ϵ)
        iter += HSD_NEWTON_CHECK_EVERY
    end
    return ws.t
end

"""
    newton_block!(ws, data, dmin, ϵ)

Run `HSD_NEWTON_CHECK_EVERY` Newton iterations of [`boxcone_root!`](@ref) without testing for
convergence. The loop is a plain one, in a function of its own so that `@trace` leaves it alone:
it is unrolled at trace time.
"""
@inline function newton_block!(ws::HSDWorkspace, data::Tuple, dmin::BatchedNumber, ϵ::Number)
    for _ in 1:HSD_NEWTON_CHECK_EVERY
        newton_residual!(ws, data)
        count_newton!(ws)
        newton_step!(ws, dmin, ϵ)
    end
    return nothing
end

"Evaluate the scalar residual of the dual (`(lc, uc)`) or primal (`(c, lv, uv)`) root at `ws.t`."
@inline function newton_residual!(ws::HSDWorkspace, (lc, uc)::NTuple{2})
    ws.g, ws.d = dual_residual!!(ws.g, ws.d, ws.by, ws.wt, ws.tt, lc, uc, ws.t)
    return nothing
end
@inline function newton_residual!(ws::HSDWorkspace, (c, lv, uv)::NTuple{3})
    ws.g, ws.d = primal_residual!!(ws.g, ws.d, ws.bx, ws.xt, ws.tt, ws.μ, ws.ηx, ws.inv_ηx, c, lv, uv, ws.t)
    return nothing
end

"Tolerance of Algorithm 1 in `T`: not below its precision, where the stopping tests could never hold."
newton_tolerance(::Type{T}) where {T} = max(convert(T, HSD_NEWTON_TOL), 2 * eps(T))

"""
    newton_init!(ws, dmin, active)

Set up Algorithm 1 from the residual `g(t)` at the starting point `t = ws.t`: since `g` has slopes
of at least `dmin`, the root lies in `[t, t - g(t)/dmin]` if `g(t) < 0` and in `[0, t]` otherwise.
This is the bracket of the paper, found without evaluating `g(0)`.
"""
@inline function newton_init!(ws::HSDWorkspace, dmin::BatchedNumber, active = true)
    # false for the running instances, typed like `ws.t`; the frozen ones are done already
    ws.done = batched_apply!!((t, a) -> (t != t) | !a, ws.done, ws.t, active)
    ws.lo = batched_apply!!(zero, ws.lo, ws.lo)
    ws.hi = batched_apply!!((t, g, dm) -> ifelse(g < zero(g), t - g / dm, t), ws.hi, ws.t, ws.g, dmin)
    ws.hi0 = copy_into!!(ws.hi0, ws.hi)
    ws.Δ = batched_apply!!(x -> oftype(x, Inf), ws.Δ, ws.Δ)
    return nothing
end

"Count one evaluation of the scalar residual for every instance still running."
@inline function count_newton!(ws::HSDWorkspace)
    ws.newton_iters = batched_apply!!((n, d) -> ifelse(d, n, n + one(n)), ws.newton_iters, ws.newton_iters, ws.done)
    return nothing
end

"""
    hsd_operator!(Tz, z, ws, hsd, step_sizes, ν)

Evaluate the PDHG operator of the embedding (eq. 28): first the dual block, a projection onto
`K[lc, uc]*` computed through Moreau's identity (eq. 34), then the primal block, the implicit
projection onto `K[lv, uv]` (eqs. 35-38). Costs one product by `A` and one by `Aᵀ`.
"""
function hsd_operator!(
        Tz::HSDPoint, z::HSDPoint, ws::HSDWorkspace, hsd::NamedTuple, step_sizes::StepSizes, ν::Number,
        active = true,
    )
    (; c, lv, uv, A, At, lc, uc) = hsd
    (; η, ω) = step_sizes
    ws.σ = batched_apply!!(*, ws.σ, η, ω)
    ws.ηx = batched_apply!!(/, ws.ηx, η, ω)
    ws.μ = batched_apply!!((λ, w) -> ν / (λ * w), ws.μ, η, ω)
    ws.dmin = batched_apply!!(inv, ws.dmin, ws.μ)
    # `1/η` for the primal root, in a scratch slot which is free during the operator
    ws.inv_ηx = batched_apply!!(inv, ws.inv_ηx, ws.ηx)

    # dual: (α⁺, y⁺) = (α̃, ỹ) + Π_{K_c}(-α̃, -ỹ), with α̃ = α - στ and ỹ = y - σAx
    Ax = mul!(ws.by, A, z.xy.x)
    σr = transpose(ws.σ)
    @. ws.wt = σr * Ax - z.xy.y
    ws.tt = batched_apply!!((α, σ, τ) -> σ * τ - α, ws.tt, z.α, ws.σ, z.τ)
    τc = boxcone_root!(ws, (lc, uc), ws.warm, one(eltype(Ax)), active)
    ws.warm = copy_into!!(ws.warm, τc)
    τcr = transpose(τc)
    @. Tz.xy.y = box_clip(ws.wt, lc, uc, τcr) - ws.wt
    Tz.α = batched_apply!!((tt, τc) -> τc - tt, Tz.α, ws.tt, τc)

    # primal: τ̃ = τ + μ(2α⁺ - α), x̃ = x + ηAᵀ(2y⁺ - y), then the root of ψ
    @. ws.by = 2Tz.xy.y - z.xy.y
    At_y = mul!(ws.bx, At, ws.by)
    ηr = transpose(ws.ηx)
    @. ws.xt = z.xy.x + ηr * At_y
    ws.tt = batched_apply!!((τ, μ, α⁺, α) -> τ + μ * (2α⁺ - α), ws.tt, z.τ, ws.μ, Tz.α, z.α)
    τp = boxcone_root!(ws, (c, lv, uv), Tz.τ, ws.dmin, active)
    Tz.τ = copy_into!!(Tz.τ, τp)
    τpr = transpose(Tz.τ)
    @. Tz.xy.x = box_clip(ws.xt - ηr * c * τpr, lv, uv, τpr)
    return Tz
end

"""
    hsd_fixed_point_error!!(dest, ws, scratch, z, Tz, A)

Compute the fixed-point residual `‖z - T(z)‖_P` in the metric `P` of the operator (eq. 24),

    Δτ²/μ + ‖Δx‖²/η + (Δα² + ‖Δy‖²)/σ - 2ΔτΔα - 2⟨AΔx, Δy⟩

column by column into `dest`, clamped below so that rounding cannot make it negative.
"""
function hsd_fixed_point_error!!(
        dest::BatchedNumber, ws::HSDWorkspace, scratch::Scratch, z::HSDPoint, Tz::HSDPoint,
        A::AbstractMatrix,
    )
    Δx = @. scratch.x = z.xy.x - Tz.xy.x
    Δy = @. scratch.y = z.xy.y - Tz.xy.y
    A_Δx = mul!(ws.by, A, Δx)
    nx = colnorm!!(scratch.b1, Δx)
    ny = colnorm!!(scratch.b2, Δy)
    ws.cross = colsum!!(ws.cross, @. ws.by = Δy * A_Δx)
    return batched_apply!!(
        dest, nx, ny, ws.cross, z.τ, Tz.τ, z.α, Tz.α, ws.μ, ws.ηx, ws.σ,
    ) do nx, ny, cross, τ, Tτ, α, Tα, μ, η, σ
        dτ, dα = τ - Tτ, α - Tα
        sq = dτ^2 / μ + nx^2 / η + (dα^2 + ny^2) / σ - 2dτ * dα - 2cross
        sqrt(max(sq, zero(sq)))
    end
end

"""
    certificate_regime!!(dest, ws, scratch, Tz, hsd)

Compute `ρ = κ⁺ / (τ + κ⁺)` at the PDHG point (eqs. 49-50), column by column into `dest`, where
`κ = -cᵀx - p(-y; lc, uc) - p(-r; lv, uv)` with the projected reduced cost `r = Π_R(cτ - Aᵀy)`.
Costs one product by `Aᵀ`.
"""
function certificate_regime!!(
        dest::BatchedNumber, ws::HSDWorkspace, scratch::Scratch, Tz::HSDPoint, hsd::NamedTuple
    )
    (; c, lv, uv, At, lc, uc) = hsd
    (; x, y) = Tz.xy
    At_y = mul!(ws.bx, At, y)
    τr = transpose(Tz.τ)
    # `bound_term` ignores the infinite bounds, so it reads `r` as projected onto `R`
    ws.κ_primal = colsum!!(ws.κ_primal, @. ws.bx = bound_term(c * τr - At_y, lv, uv) - c * x)
    ws.κ_dual = colsum!!(ws.κ_dual, @. scratch.y = bound_term(y, lc, uc))
    return batched_apply!!(dest, ws.κ_primal, ws.κ_dual, Tz.τ) do a, b, τ
        κ = max(a + b, zero(a))
        ifelse(τ + κ > zero(τ), κ / (τ + κ), zero(τ))
    end
end

"""
    hsd_restart!(state, algo)

Restart the instances selected by [`halpern_restart_check!`](@ref): update their primal weight
with the PI controller of the paper (§5.3), then move both their iterate and their anchor to the
PDHG point `T(z)`, rescaled as in eq. 44. The other instances are left untouched.
"""
function hsd_restart!(state::HSDPDLPState{T}, algo::Algorithm{:HSDPDLP}) where {T}
    (; Tz, z, z_anchor, hsd, ws, step_sizes, scratch, halpern) = state
    (; restart) = halpern
    hsd_primal_weight_update!(halpern, step_sizes, ws, scratch, Tz, z_anchor, hsd, algo.step_size)
    state.stats.extra_products += 1  # the `Aᵀy` of `certificate_regime!!`

    # homogeneous rescaling of the restart point (eq. 44)
    nx = colnorm!!(scratch.b1, Tz.xy.x)
    ny = colnorm!!(scratch.b2, Tz.xy.y)
    ws.restart_scale = batched_apply!!(ws.restart_scale, nx, ny, Tz.α, Tz.τ) do nx, ny, α, τ
        h = max(nx, ny, abs(α))
        g = max(h, τ)
        extreme = (g < oftype(g, 1.0e-12)) | (g > oftype(g, 1.0e12))
        normalize = (τ > oftype(τ, 1.0e-8) * h) & ((τ < oftype(τ, 0.1)) | (τ > oftype(τ, 10)))
        ifelse(extreme, inv(g), ifelse(normalize, inv(τ), one(τ)))
    end
    hsd_restart_point!(z_anchor, restart, ws.restart_scale, Tz)
    hsd_restart_point!(z, restart, ws.restart_scale, Tz)
    # `T` is positively homogeneous, so both scalar roots of the next step scale with the
    # point: the warm starts follow (`Tz` is overwritten by the next step before it is read)
    rescale(r, s, v) = ifelse(r, s * v, v)
    ws.warm = batched_apply!!(rescale, ws.warm, restart, ws.restart_scale, ws.warm)
    Tz.τ = batched_apply!!(rescale, Tz.τ, restart, ws.restart_scale, Tz.τ)

    halpern.inner = batched_apply!!(halpern.inner, restart, halpern.inner) do r, k
        ifelse(r, zero(k), k)
    end
    halpern.restarts = batched_apply!!(halpern.restarts, restart, halpern.restarts) do r, n
        ifelse(r, n + 1, n)
    end
    halpern.fpe_pending = batched_apply!!(r -> r & true, halpern.fpe_pending, restart)
    return nothing
end

"Overwrite the instances of `p` which restart with `scale * Tz`."
function hsd_restart_point!(p::HSDPoint, restart::BatchedNumber, scale::BatchedNumber, Tz::HSDPoint)
    rr, sr = transpose(restart), transpose(scale)
    @. p.xy.x = ifelse(rr, sr * Tz.xy.x, p.xy.x)
    @. p.xy.y = ifelse(rr, sr * Tz.xy.y, p.xy.y)
    p.τ = batched_apply!!((r, s, t, v) -> ifelse(r, s * t, v), p.τ, restart, scale, Tz.τ, p.τ)
    p.α = batched_apply!!((r, s, t, v) -> ifelse(r, s * t, v), p.α, restart, scale, Tz.α, p.α)
    return p
end

"""
    hsd_primal_weight_update!(halpern, step_sizes, ws, scratch, Tz, z_anchor, hsd, params)

PI control of the primal weight (eqs. 46-52) for the instances which restart: the error is
`e = log(δd / (ω δp))` with `δp² = ‖Δx‖² + (λ²/ν) Δτ²` and `δd² = ‖Δy‖² + Δα²` the movements of
the primal and dual blocks since the anchor, the integral `s ← ζ s + e` and the weight
`ω ← ω exp(kP e + kI s)`. The weight is held, and the integral reset, in the certificate regime
`ρ > 0.9`; it is held, with the integral, when a movement vanishes.
"""
function hsd_primal_weight_update!(
        halpern::HalpernStats, step_sizes::StepSizes, ws::HSDWorkspace, scratch::Scratch,
        Tz::HSDPoint, z_anchor::HSDPoint, hsd::NamedTuple, params::StepSizeParameters,
    )
    (; restart) = halpern
    (; pid_kp, pid_ki, pid_integral_discount, invnorm_scaling) = params
    ws.ρ = certificate_regime!!(ws.ρ, ws, scratch, Tz, hsd)
    ρ = ws.ρ
    Δx = colnorm!!(scratch.b1, @. scratch.x = Tz.xy.x - z_anchor.xy.x)
    Δy = colnorm!!(scratch.b2, @. scratch.y = Tz.xy.y - z_anchor.xy.y)
    ν = invnorm_scaling
    # δp, then δd, into the per-instance scratch
    ws.δp = batched_apply!!(ws.δp, Δx, Tz.τ, z_anchor.τ, step_sizes.η) do Δx, τ, τ0, λ
        sqrt(Δx^2 + λ^2 / ν * (τ - τ0)^2)
    end
    ws.δd = batched_apply!!((Δy, α, α0) -> sqrt(Δy^2 + (α - α0)^2), ws.δd, Δy, Tz.α, z_anchor.α)
    valid(δp, δd) = (δp > zero(δp)) & (δd > zero(δd)) & isfinite(δp) & isfinite(δd)
    safelog(a) = log(ifelse(a > zero(a), a, one(a)))
    halpern.pid_error = batched_apply!!(halpern.pid_error, ws.δp, ws.δd, step_sizes.ω) do δp, δd, w
        safelog(δd) - safelog(δp) - log(w)
    end
    gate = oftype(ν, 0.9)
    # safeguard (not in the paper): the weight stays within a factor `HSD_OMEGA_WINDOW` of its
    # initial value 1, and the integral stops accumulating while the proposed weight is outside
    W = oftype(ν, HSD_OMEGA_WINDOW)
    proposal(w, e, s) = w * exp(pid_kp * e + pid_ki * (pid_integral_discount * s + e))
    inside(w) = (w >= inv(W)) & (w <= W)
    halpern.pid_error_sum = batched_apply!!(
        halpern.pid_error_sum, restart, ρ, ws.δp, ws.δd, step_sizes.ω, halpern.pid_error, halpern.pid_error_sum,
    ) do r, ρ, δp, δd, w, e, s
        accumulate = valid(δp, δd) & inside(proposal(w, e, s))
        ifelse(r, ifelse(ρ > gate, zero(s), ifelse(accumulate, pid_integral_discount * s + e, s)), s)
    end
    # the integral is already updated, so the proposal is recomputed from the error alone
    step_sizes.ω = batched_apply!!(
        step_sizes.ω, restart, ρ, ws.δp, ws.δd, step_sizes.ω, halpern.pid_error, halpern.pid_error_sum,
    ) do r, ρ, δp, δd, w, e, s_new
        update = r & (ρ <= gate) & valid(δp, δd)
        ifelse(update, clamp(w * exp(pid_kp * e + pid_ki * s_new), inv(W), W), w)
    end
    return nothing
end
