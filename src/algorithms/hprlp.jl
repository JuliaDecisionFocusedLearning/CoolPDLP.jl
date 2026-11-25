"""
    HPRLP(args...; kwargs...)

Shortcut for [`Algorithm{:HPRLP}`](@ref): the Halpern Peaceman-Rachford method with semi-proximal
terms of HPR-LP (Chen, Sun, Yuan, Zhang and Zhao, "HPR-LP: An implementation of an HPR
method for solving linear programming", Mathematical Programming Computation, 2025,
[doi:10.1007/s12532-025-00292-0](https://doi.org/10.1007/s12532-025-00292-0)), for linear
programs only.

With `T₁ = λI - AAᵀ` and `λ ≥ λ_max(AAᵀ)`, a step of HPR-LP is a PDHG step with primal step `σ`
and dual step `1 / (λσ)`, reflected and anchored as in [`HPDLP`](@ref). What sets it apart is
everything around the step:

- `λ` is `1.01` times the estimate of a power iteration on `AAᵀ` (`5000` iterations at most,
  tolerance `1e-4` on the residual, checked every `10` iterations, same random start), and grows
  on the fly if the restart metric ever comes out negative;
- restarts are measured with the fixed-point residual in the metric `M` of the splitting
  (see [`hprlp_fixed_point_error!`](@ref)), the first check always restarts, and the decays are
  `0.2` (sufficient), `0.6` (necessary) and `0.2` (artificial), with a check every `150` iterations;
- at each restart, the penalty `σ` moves to `‖Δx‖ / (‖Δy‖ √λ)` blended in log space with the
  `σ` of the smallest residual so far (see [`hprlp_sigma`](@ref));
- the problem is scaled by `20` Curtis-Reid iterations, `10` Ruiz passes, Pock-Chambolle with
  `α = 1`, then by `1 / (1 + ‖b‖)` and `1 / (1 + ‖c‖)`, and `σ` starts at `‖b‖ / ‖c‖` of the
  scaled problem;
- the termination test reads the reduced costs `z̄` of the splitting (see [`kkt_errors!`](@ref)).

Termination is only checked every `check_every` iterations, as in the paper.

Every instance of a batch restarts, and updates its `σ` and `λ`, on its own.
"""
function HPRLP(args...; kwargs...)
    return Algorithm{:HPRLP}(
        args...;
        curtis_reid_iter = 20,
        ruiz_iter = 10,
        chambolle_pock_alpha = 1.0,
        bound_objective_rescaling = true,
        sufficient_decay = 0.2,
        necessary_decay = 0.6,
        artificial_decay = 0.2,
        check_every = 150,
        kwargs...
    )
end

"""
    HPRLPState

The step sizes are those of PDHG, `η = 1 / √λ` and `ω = η / σ`, so that the primal step `η / ω`
is the penalty `σ` of HPR-LP and the dual step `η ω` is `1 / (λσ)`.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct HPRLPState{
        T <: Number,
        V <: AbstractVecOrMat{T},
        SS <: StepSizes,
        Sc <: Scratch,
        HS <: HalpernStats,
        CS <: ConvergenceStats,
        IW <: Union{Nothing, InfeasibilityWorkspace},
        B <: BatchedNumber,
    } <: AbstractState{T, V}
    "image `w̄ = (x̄, ȳ)` of the Halpern iterate by the last step, which is the reported solution"
    sol::PrimalDualSolution{T, V}
    "current Halpern iterate"
    z::PrimalDualSolution{T, V}
    "Halpern anchor, set at the last restart"
    z_anchor::PrimalDualSolution{T, V}
    "reduced costs `z̄` of the last step before a check, read by the termination test"
    z_bar::V
    "step sizes"
    step_sizes::SS
    "upper bound `λ` on the largest eigenvalue of `AAᵀ`, per instance since it may grow"
    λ::B
    "penalty `σ` in use when the smallest fixed-point residual so far was measured"
    best_σ::B
    "smallest fixed-point residual measured at a check so far"
    best_fpe::B
    "per-column scratch"
    t1::B
    "per-column scratch"
    t2::B
    "scratch space"
    scratch::Sc
    "Halpern and restart bookkeeping (its PID fields are unused)"
    halpern::HS
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
end

nbinstances((; sol)::HPRLPState) = nbinstances(sol)
function instance(state::HPRLPState, i::Int)
    return HPRLPState(
        instance(state.sol, i),
        instance(state.z, i),
        instance(state.z_anchor, i),
        instance_vec(state.z_bar, i),
        instance(state.step_sizes, i),
        instance_num(state.λ, i),
        instance_num(state.best_σ, i),
        instance_num(state.best_fpe, i),
        instance_num(state.t1, i),
        instance_num(state.t2, i),
        instance(state.scratch, i),
        instance(state.halpern, i),
        instance(state.stats, i),
        instance(state.infeas, i),
    )
end

"""
    hprlp_power_iteration(A, At; maxiter = 5000, tol = 1e-4, check_every = 10)

Estimate the largest eigenvalue of `AAᵀ` by a power iteration:
start from `randn(MersenneTwister(1), m) .+ 1e-8`, normalize and multiply by `AAᵀ`, and every
`check_every` iterations stop if `‖AAᵀq - λq‖ < tol` with the Rayleigh quotient `λ`.

It deliberately duplicates [`spectral_norm`](@ref) to keep this exact starting point and
stopping rule. The tolerance is absolute: it only makes sense because the problem is
already scaled (so that `‖A‖` is of order one) when this runs.
"""
function hprlp_power_iteration(
        A::AbstractMatrix{T}, At::AbstractMatrix{T};
        maxiter::Integer = 5000, tol::Number = 1.0e-4, check_every::Integer = 10,
    ) where {T}
    m, n = size(A)
    backend = get_backend(A)
    z_cpu = randn(Random.MersenneTwister(1), m) .+ 1.0e-8
    z = copyto!(allocate(backend, T, m), convert(Vector{T}, z_cpu))
    q = allocate(backend, T, m)
    Atq = allocate(backend, T, n)
    λ = one(T)
    for i in 1:maxiter
        invn = inv(sqrt(dot(z, z) + eps(T)))
        @. q = z * invn
        mul!(Atq, At, q)
        mul!(z, A, Atq)
        if i % check_every == 0
            λ = dot(q, z)
            @. q = z - λ * q
            norm(q) < tol && return λ
        end
    end
    return λ
end

"""
    hprlp_initial_sigma(milp)

Initial penalty: `‖b‖ / ‖c‖` on the scaled problem, where
`bᵢ` is the largest finite absolute bound of row `i`, or `1` if either norm is below `1e-8`.
"""
function hprlp_initial_sigma(milp::MILP{T}) where {T}
    (; c, lc, uc) = milp
    b_norm = colnorm(combine.(lc, uc))
    c_norm = colnorm(c)
    tol = convert(T, 1.0e-8)
    return broadcast(b_norm, c_norm) do b, c
        ifelse((b > tol) & (c > tol), b / c, one(T))
    end
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:HPRLP};
        starting_time::Float64
    ) where {T, V}
    λ_est = hprlp_power_iteration(milp.A, milp.At)
    # without any constraint, `λ` plays no role and any positive value will do
    λ0 = ifelse(λ_est > zero(T), convert(T, 1.01) * λ_est, one(T))
    λ = batched_expand(sol.x, λ0)
    σ0 = batched_expand(sol.x, hprlp_initial_sigma(milp))
    η = batched_apply!!(λ -> inv(sqrt(λ)), batched_similar(λ), λ)
    ω = batched_apply!!((η, σ) -> η / σ, batched_similar(λ), η, σ0)
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    halpern = HalpernStats(sol, ω)
    # the first check restarts no matter what: with no reference residual (`Inf`),
    # the sufficient decay criterion always holds
    halpern.fpe_pending = batched_apply!!(zero, halpern.fpe_pending, halpern.fpe_pending)
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp, algo.generic.termination_norm)
    tol = algo.termination.infeasibility_tol
    stats = ConvergenceStats(err; ray = ray_buffer(sol, tol), starting_time, extra_products = 2)
    return HPRLPState(;
        sol, z = copy(sol), z_anchor = copy(sol), z_bar = zero(sol.x),
        step_sizes, λ, best_σ = copy(σ0),
        best_fpe = batched_expand(sol.x, convert(T, Inf)),
        t1 = batched_expand(sol.x, zero(T)), t2 = batched_expand(sol.x, zero(T)),
        scratch, halpern, stats, infeas = InfeasibilityWorkspace(sol, tol),
    )
end

"""
    solve!(state::HPRLPState, milp, algo)

Run HPR-LP until termination, with the same structure of checks as [`HPDLP`](@ref): each block
of `check_every` steps measures the fixed-point residual on its first step (the reference
residual of the instances which just restarted) and on its last one, then checks termination,
decides the restarts and applies them along with the update of `σ`.
"""
function solve!(state::HPRLPState, milp::MILP, algo::Algorithm{:HPRLP})
    check_every = algo.generic.check_every
    check_every >= 2 || throw(ArgumentError("HPRLP needs `check_every >= 2`, got $check_every"))
    prog = init_progress("HPRLP iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        must_terminate = check_pass!(state, milp, algo, prog)
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    check_pass!(state::HPRLPState, milp, algo, prog = NoProgress())

Run one block of `check_every` steps, the termination check, the restarts and the update of `σ`,
and return whether to stop (see [`solve!`](@ref)).
"""
function check_pass!(
        state::HPRLPState,
        milp::MILP,
        algo::Algorithm{:HPRLP},
        prog = NoProgress()
    )
    step!(state, milp, algo, Val(:first))
    seed_fixed_point_error!(state.halpern)
    next_progress!(prog, state)
    @trace for _ in 1:(algo.generic.check_every - 2)
        step!(state, milp, algo, Val(:inner))
        next_progress!(prog, state)
    end
    step!(state, milp, algo, Val(:last))
    next_progress!(prog, state)
    must_terminate = hprlp_termination_check!(
        state.stats, state.infeas, state.scratch, state.sol, state.z_bar, state.z_anchor,
        milp, algo.termination, algo.generic,
    )
    halpern_restart_check!(state.halpern, algo.restart, must_terminate, active_mask(state.stats))
    restart!(state, algo)
    return must_terminate
end

"See [`ray_candidates`](@ref) for HPDLP."
ray_candidates(state::HPRLPState) = ((state.sol, state.z_anchor), (state.sol, nothing))

"""
    hprlp_termination_check!(stats, infeas, scratch, sol, z_bar, z_anchor, milp, params, generic)

Termination test of HPR-LP: the usual relative KKT errors of `sol = (x̄, ȳ)`, except that the
dual residual and the dual objective use the reduced costs `z̄` of the splitting. The
infeasibility candidates are those of [`HPDLP`](@ref).
"""
function hprlp_termination_check!(
        stats::ConvergenceStats, infeas::Union{Nothing, InfeasibilityWorkspace}, scratch::Scratch,
        sol::PrimalDualSolution, z_bar::AbstractVecOrMat, z_anchor::PrimalDualSolution,
        milp::MILP, params::TerminationParameters, generic::GenericParameters,
    )
    update_time_elapsed!(stats, generic.time_limit)
    kkt_errors!(stats.err, scratch, sol, z_bar, milp, generic.termination_norm)
    stats.extra_products += 2
    infeasibility_check!(
        stats, infeas, scratch, (sol, z_anchor), (sol, nothing), milp, params.infeasibility_tol,
        active_mask(stats),
    )
    record_error_history!(stats, generic.record_error_history)
    return set_termination_status!!(stats, scratch.b1, params, generic.time_limit)
end

"""
    step!(state::HPRLPState, milp, algo, Val(kind))

Apply one step of HPR-LP, with `σ = η / ω` and `λσ = 1 / (η ω)`:

    x̄ = proj_[lv, uv](x + σ (Aᵀy - c))
    x̂ = 2x̄ - x
    v = A x̂ - λσ y
    ȳ = (proj_[lc, uc](v) - v) / (λσ)

followed by the reflected Halpern combination of `(x, y)` with `(x̄, ȳ)` and the anchor. `kind`
is `:inner` for a plain step, `:first` for the first step of a block, which also measures the
fixed-point residual, and `:last` for the last one, which measures it as well and forms the
reduced costs `z̄ = (x̄ - (x + σ (Aᵀy - c))) / σ` for the termination test.
"""
function step!(
        state::HPRLPState{T}, milp::MILP{T}, algo::Algorithm{:HPRLP}, ::Val{kind},
    ) where {T, kind}
    (; sol, z, z_anchor, z_bar, step_sizes, scratch, halpern) = state
    (; c, lv, uv, A, At, lc, uc) = milp
    (; η, ω) = step_sizes
    σ = transpose(batched_apply!!(/, scratch.b1, η, ω))
    σd = transpose(batched_apply!!(*, scratch.b2, η, ω))

    # a frozen instance of a batch keeps its `(x̄, ȳ)`, `z̄` and `z` (see `active_mask`)
    active = active_mask(state.stats)
    At_y = mul!(scratch.z, At, z.y)
    hprlp_primal_step!(sol.x, z_bar, z.x, σ, c, At_y, lv, uv, Val(kind === :last), active)
    x_hat = @. scratch.x = 2 * sol.x - z.x
    A_x_hat = mul!(scratch.y, A, x_hat)
    # `v = A x̂ - y / σd` is formed twice rather than stored, to keep a single kernel
    masked_broadcast!(hprlp_dual_step, sol.y, active, A_x_hat, z.y, σd, lc, uc)

    if kind === :first
        # only the first step of a cycle after a restart is measured
        hprlp_fixed_point_error!(state, milp, halpern.fpe_pending)
        state.stats.extra_products += 1
    elseif kind === :last
        hprlp_fixed_point_error!(state, milp, true)
        state.stats.extra_products += 1
    end
    γ = algo.step_size.reflection_coefficient
    w = batched_apply!!(k -> (k + 1) / (k + 2), scratch.b1, halpern.inner)
    halpern_combine!(z.x, w, γ, sol.x, z_anchor.x, active)
    halpern_combine!(z.y, w, γ, sol.y, z_anchor.y, active)
    halpern.inner = masked_apply!!(k -> k + 1, halpern.inner, active, halpern.inner)
    halpern.total += 1
    state.stats.kkt_passes += 1
    return nothing
end

"""
    hprlp_primal_step!(x_bar, z_bar, x, σ, c, At_y, lv, uv, Val(reduced_costs))

Write `x̄ = proj_[lv, uv](x + σ (Aᵀy - c))` into `x_bar`, and with `Val(true)` also the reduced
costs `z̄ = (x̄ - (x + σ (Aᵀy - c))) / σ` into `z_bar`, from the very value that was projected, so
that `z̄` is exactly zero wherever no bound is active.
"""
function hprlp_primal_step!(x_bar, z_bar, x, σ, c, At_y, lv, uv, ::Val{false}, active)
    masked_broadcast!(hprlp_primal_point, x_bar, active, x, σ, c, At_y, lv, uv)
    return nothing
end
function hprlp_primal_step!(x_bar, z_bar, x, σ, c, At_y, lv, uv, ::Val{true}, active)
    masked_broadcast!(hprlp_primal_point, x_bar, active, x, σ, c, At_y, lv, uv)
    masked_broadcast!(z_bar, active, x_bar, x, σ, c, At_y) do xb, x, σ, c, a
        (xb - (x - σ * (c - a))) / σ
    end
    return nothing
end

"Primal point `x̄ = proj_[lv, uv](x - σ (c - Aᵀy))` of one entry."
@inline hprlp_primal_point(x, σ, c, a, l, u) = clamp(x - σ * (c - a), l, u)

"Dual point `ȳ = σd (proj_[lc, uc](v) - v)` of one entry, with `v = A x̂ - y / σd`."
@inline hprlp_dual_step(Ax, y, σd, l, u) = σd * (clamp(Ax - y / σd, l, u) - (Ax - y / σd))

"""
    hprlp_fixed_point_error!(state, milp, allow_growth)

Measure the fixed-point residual of the last step in the metric `M` of the
splitting, column by column, into
`state.halpern.fpe`. With `Δ = z - T(z)` and `σ` the primal step:

    ‖Δ‖²_M = σλ ‖Δy‖² + ‖Δx‖² / σ + 2 ⟨A Δx, Δy⟩

If this comes out negative, `λ` underestimates the largest eigenvalue of `AAᵀ`, and it is raised
to `-(‖Δx‖² / σ + 2 ⟨A Δx, Δy⟩) / (σ ‖Δy‖²) × 1.05`, which also changes the steps; the residual
is then `√(-(‖Δx‖² / σ + 2 ⟨A Δx, Δy⟩) × 0.05)`. `λ` only grows where `allow_growth` holds, which
is the case whenever the residual is measured, and never for a frozen
instance of a batch. Costs one product by `A`.
"""
function hprlp_fixed_point_error!(state::HPRLPState, milp::MILP, allow_growth::BatchedNumber)
    (; sol, z, step_sizes, scratch, halpern) = state
    (; A) = milp
    # a frozen instance of a batch keeps its `λ` and steps: its `z - T(z)` no longer means anything
    active = active_mask(state.stats)
    (; η, ω) = step_sizes
    Δy_norm = colnorm!!(scratch.b2, @. scratch.y = z.y - sol.y)
    Δx = @. scratch.x = z.x - sol.x
    Δx_norm = colnorm!!(scratch.b1, Δx)
    # ‖Δx‖² / σ and ‖Δy‖²
    state.t1 = batched_apply!!((n, η, ω) -> n^2 * ω / η, state.t1, Δx_norm, η, ω)
    state.t2 = batched_apply!!(n -> n^2, state.t2, Δy_norm)
    A_Δx = mul!(scratch.y, A, Δx)
    cross = colsum!!(scratch.b1, @. A_Δx = A_Δx * (z.y - sol.y))
    # the negative case is computed first, while `λ` is still the old one
    halpern.fpe = batched_apply!!(
        halpern.fpe, state.t1, state.t2, cross, state.λ, η, ω, allow_growth, active,
    ) do a, b, x, λ, η, ω, g, act
        rest = a + 2x
        square = η / ω * λ * b + rest
        # both branches are evaluated, so both square roots are guarded
        grown = sqrt(max(-rest * convert(typeof(a), 0.05), zero(rest)))
        ifelse(g & act & (square < 0), grown, sqrt(max(square, zero(square))))
    end
    state.λ = batched_apply!!(
        state.λ, state.t1, state.t2, cross, state.λ, η, ω, allow_growth, active,
    ) do a, b, x, λ, η, ω, g, act
        rest = a + 2x
        square = η / ω * λ * b + rest
        ifelse(g & act & (square < 0), -rest / (η / ω * b) * convert(typeof(a), 1.05), λ)
    end
    # the penalty `σ = η / ω` is kept, the steps follow the new `λ`
    state.t1 = batched_apply!!((η, ω) -> η / ω, state.t1, η, ω)
    step_sizes.η = batched_apply!!(λ -> inv(sqrt(λ)), step_sizes.η, state.λ)
    step_sizes.ω = batched_apply!!((η, σ) -> η / σ, step_sizes.ω, step_sizes.η, state.t1)
    return nothing
end

"""
    hprlp_sigma(restart, Δx, Δy, λ, σ, fpe, best_fpe, best_σ, rel_primal, rel_dual, rel_gap)

New penalty of one instance at a check: unchanged
without a restart; otherwise, if `Δx = ‖x̄ - x⁰‖` and `Δy = ‖ȳ - y⁰‖` both lie in `(1e-16, 1e12)`,

    ratio = Δx / (Δy √λ)
    fact = exp(-0.05 fpe / best_fpe)
    σ = κ exp(fact log(ratio) + (1 - fact) log(best_σ))

where `κ` is `1` unless `max(min(rel_dual, rel_primal), min(rel_gap, fpe))` is below `9e-10`, in
which case it is `rel_dual / rel_primal` (its square root below `9e-10` but above `5e-10`),
clamped to `[1e-2, 1e2]`. Otherwise `σ = 1`.
"""
function hprlp_sigma(r, Δx, Δy, λ, σ, fpe, best_fpe, best_σ, p, d, g)
    T = typeof(σ)
    valid = (Δx > T(1.0e-16)) & (Δy > T(1.0e-16)) & (Δx < T(1.0e12)) & (Δy < T(1.0e12))
    # the logarithms are guarded so that a degenerate instance gives a finite, unused value
    ratio = ifelse(valid, Δx / Δy / sqrt(λ), one(T))
    # an exactly converged, non-terminated instance has `fpe = best_fpe = 0` (with
    # `termination_reltol = 0`): keep the estimate instead of a NaN from `0 / 0`
    fact = ifelse(best_fpe > zero(best_fpe), exp(-T(0.05) * fpe / best_fpe), one(T))
    candidate = exp(fact * log(ratio) + (1 - fact) * log(best_σ))
    q = d / p
    temp = max(min(d, p), min(g, fpe))
    κ = ifelse(
        temp > T(9.0e-10),
        one(T),
        ifelse(temp > T(5.0e-10), clamp(sqrt(q), T(1.0e-2), T(1.0e2)), clamp(q, T(1.0e-2), T(1.0e2))),
    )
    return ifelse(r, ifelse(valid, κ * candidate, one(T)), σ)
end

"""
    restart!(state::HPRLPState, algo)

At a check: keep track of the penalty `σ` of the smallest fixed-point residual so far (whether
or not the instance restarts), then for the instances selected by
[`halpern_restart_check!`](@ref), update `σ` with [`hprlp_sigma`](@ref) and move both the
iterate and the anchor to `T(z)`. Every update is masked column by column.
"""
function restart!(state::HPRLPState{T}, algo::Algorithm{:HPRLP}) where {T}
    (; sol, z, z_anchor, step_sizes, scratch, halpern, stats) = state
    (; restart, fpe) = halpern
    (; η, ω) = step_sizes
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = stats.err
    σ = batched_apply!!((η, ω) -> η / ω, scratch.b1, η, ω)
    state.best_σ = batched_apply!!(state.best_σ, fpe, state.best_fpe, σ, state.best_σ) do f, b, s, bs
        ifelse(f < b, s, bs)
    end
    state.best_fpe = batched_apply!!(min, state.best_fpe, state.best_fpe, fpe)

    state.t1 = colnorm!!(state.t1, @. scratch.x = sol.x - z_anchor.x)
    state.t2 = colnorm!!(state.t2, @. scratch.y = sol.y - z_anchor.y)
    σ_new = batched_apply!!(
        scratch.b1, restart, state.t1, state.t2, state.λ, σ, fpe, state.best_fpe, state.best_σ,
        primal, primal_scale, dual, dual_scale, gap, gap_scale,
    ) do r, Δx, Δy, λ, σ, f, bf, bs, p, ps, d, ds, g, gs
        hprlp_sigma(r, Δx, Δy, λ, σ, f, bf, bs, p / ps, d / ds, g / gs)
    end
    step_sizes.ω = batched_apply!!((η, σ) -> η / σ, step_sizes.ω, η, σ_new)

    batched_select!(z_anchor, restart, sol)
    batched_select!(z, restart, sol)
    halpern.inner = batched_apply!!(halpern.inner, restart, halpern.inner) do r, k
        ifelse(r, zero(k), k)
    end
    halpern.restarts = batched_apply!!(halpern.restarts, restart, halpern.restarts) do r, n
        ifelse(r, n + 1, n)
    end
    # a new value, not `restart` itself (see `restart!` of HPDLP)
    halpern.fpe_pending = batched_apply!!(r -> r & true, halpern.fpe_pending, restart)
    return nothing
end
