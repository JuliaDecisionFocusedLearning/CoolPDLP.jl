using CoolPDLP
using CoolPDLP: initialize, instance, restart!, restart_check!, step!, termination_status
using JLArrays
using KernelAbstractions
using LinearAlgebra: dot
using MathOptBenchmarkInstances
using JuMP: MOI
using Random
using SparseArrays
using Test

@testset "Defaults" begin
    algo = HPDLP()
    @test algo isa CoolPDLP.Algorithm{:HPDLP}
    @test algo.step_size.invnorm_scaling == 0.998
    @test algo.step_size.reflection_coefficient == 1.0
    @test algo.step_size.pid_kp == 0.99
    @test algo.step_size.pid_ki == 0.01
    @test algo.step_size.pid_kd == 0.0
    @test algo.step_size.pid_integral_discount == 0.3
    @test algo.restart.sufficient_decay == 0.2
    @test algo.restart.necessary_decay == 0.5
    @test algo.restart.artificial_decay == 0.36
    @test algo.generic.check_every == 200
    # PDLP keeps its own defaults
    @test PDLP().restart.necessary_decay == 0.8
    @test PDLP().step_size.invnorm_scaling == 0.9
    @test occursin("HPDLP algorithm", sprint(show, algo))
    @test_throws ArgumentError solve(random_milp_batch(Xoshiro(0), 5, 8, 0.5, 1)[1][1], HPDLP(; check_every = 1))
end

@testset "Convergence on Netlib" begin
    names = ["afiro", "adlittle", "blend", "sc50a"]
    configs = [(SparseMatrixCSC, CPU()), (CoolPDLP.GPUSparseMatrixCSR, JLBackend())]
    @testset "$name on $M" for name in names, (M, backend) in configs
        milp = MILP(read_instance(Netlib, name)[1]; dataset = Netlib, name)
        algo = HPDLP(Float64, Int, M; backend, termination_reltol = 1.0e-6, max_kkt_passes = 10^6)
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        @test is_feasible(Array(sol.x), milp; cons_tol = 1.0e-3)
        sol_ref, _ = solve(milp, PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6))
        @test objective_value(Array(sol.x), milp) ≈ objective_value(sol_ref.x, milp) rtol = 1.0e-4
    end
end

@testset "Halpern step" begin
    # a single step from the anchor, with γ = 1/2, is the averaged PDHG step
    milp, sol = CoolPDLP.random_milp_and_sol(Xoshiro(1), 10, 20, 0.4)
    algo = HPDLP(; reflection_coefficient = 0.5)
    state = initialize(milp, copy(sol), algo; starting_time = time())
    z0 = copy(state.z)
    step!(state, milp, algo, Val(false))
    # k = 0: z₁ = 1/2 (1/2 (2T(z₀) - z₀) + 1/2 z₀) + 1/2 z₀ = 1/2 T(z₀) + 1/2 z₀
    @test state.z.x ≈ (state.sol.x + z0.x) / 2
    @test state.z.y ≈ (state.sol.y + z0.y) / 2
    @test state.halpern.inner == 1
end

@testset "Fixed-point residual is the norm of PDHG's metric" begin
    # firm nonexpansiveness of the PDHG operator in that norm pins down the metric, including the
    # sign of its cross term: ‖Tz - Tw‖² + ‖(z - Tz) - (w - Tw)‖² ≤ ‖z - w‖²
    rng = Xoshiro(3)
    milp, sol = CoolPDLP.random_milp_and_sol(rng, 10, 20, 0.4)
    algo = HPDLP()
    state = initialize(milp, copy(sol), algo; starting_time = time())
    (; scratch, step_sizes) = state
    step_sizes.ω = 1.7
    T(z) = CoolPDLP.pdhg_update!(copy(z), scratch, step_sizes, milp, true)
    dist(a, b) = CoolPDLP.fixed_point_error!!(0.0, scratch, a, b, milp, step_sizes)
    randpoint() = PrimalDualSolution(randn(rng, length(sol.x)), randn(rng, length(sol.y)))
    for _ in 1:50
        z, w = randpoint(), randpoint()
        Tz, Tw = T(z), T(w)
        shifted = PrimalDualSolution(z.x - Tz.x + Tw.x, z.y - Tz.y + Tw.y)
        @test dist(Tz, Tw)^2 + dist(shifted, w)^2 <= dist(z, w)^2 + 1.0e-9
    end
    # by hand: η (‖Δx‖²/τ + ‖Δy‖²/σ + 2⟨Δx, AᵀΔy⟩) with τ = η/ω and σ = ηω
    z, w = randpoint(), randpoint()
    η, ω = step_sizes.η, step_sizes.ω
    Δx, Δy = z.x - w.x, z.y - w.y
    @test dist(z, w)^2 ≈ ω * sum(abs2, Δx) + sum(abs2, Δy) / ω + 2η * dot(Δx, milp.At * Δy)
end

@testset "Restart behavior" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    # an unreachable tolerance, so that the solve runs into the iteration limit
    algo = HPDLP(; check_every = 20, termination_reltol = 0.0, max_kkt_passes = 4000)
    milp_p, sol_p = preprocess(milp, PrimalDualSolution(milp), algo)
    state = initialize(milp_p, sol_p, algo; starting_time = time())
    (; halpern) = state
    ω0 = state.step_sizes.ω

    for _ in 1:algo.generic.check_every
        step!(state, milp_p, algo, Val(true))
    end
    # the first check always restarts: the inner loop is the whole run so far
    @test restart_check!(state, algo, false)
    # unless the solver terminates
    @test !restart_check!(state, algo, true)
    @test restart_check!(state, algo, false)
    restart!(state, milp_p, algo)
    @test halpern.inner == 0
    @test halpern.restarts == 1
    @test halpern.fpe_last == Inf
    @test halpern.fpe_pending
    # both the iterate and the anchor move to T(z)
    @test state.z.x == state.sol.x == state.z_anchor.x
    @test state.z.y == state.sol.y == state.z_anchor.y
    @test state.step_sizes.ω != ω0

    state = initialize(milp_p, sol_p, algo; starting_time = time())
    solve!(state, milp_p, algo)
    @test termination_status(state.stats) == MOI.ITERATION_LIMIT
    # adaptive restarts happen, but not at every check
    @test 1 < state.halpern.restarts < state.stats.kkt_passes / algo.generic.check_every
    @test isfinite(state.halpern.fpe_initial)
    @test all(isfinite, state.sol.x)
end

@testset "Batch matches per-instance solves" begin
    # one Netlib instance with a rescaled objective per column, which keeps every instance
    # well-conditioned enough for the batched and single runs to stay comparable
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    scales = [1.0, 1.3, 0.7]
    nbatch = length(scales)
    rep(v) = repeat(v, 1, nbatch)
    milp_batch = MILP(;
        c = stack(scale * milp.c for scale in scales), lv = rep(milp.lv), uv = rep(milp.uv),
        milp.A, lc = rep(milp.lc), uc = rep(milp.uc), milp.int_var,
    )
    milps = [instance(milp_batch, i) for i in 1:nbatch]
    rng = Xoshiro(1)
    xs = [randn(rng, nbvar(milp_batch)) for _ in 1:nbatch]
    ys = [randn(rng, nbcons(milp_batch)) for _ in 1:nbatch]
    sols = map(PrimalDualSolution, xs, ys)
    sol_batch = PrimalDualSolution(stack(xs), stack(ys))
    # an unreachable tolerance, so that every solve runs for the same number of iterations
    algo = HPDLP(; check_every = 10, termination_reltol = 0.0, max_kkt_passes = 400)

    state_batch = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
    solve!(state_batch, milp_batch, algo)
    states = map(1:nbatch) do i
        state = initialize(milps[i], copy(sols[i]), algo; starting_time = time())
        solve!(state, milps[i], algo)
    end
    @test allunique(eachcol(state_batch.sol.x))
    for i in 1:nbatch
        state_i = instance(state_batch, i)
        @test state_i.stats.kkt_passes == states[i].stats.kkt_passes
        @test state_i.sol ≈ states[i].sol
        @test state_i.z ≈ states[i].z
        @test state_i.step_sizes.ω ≈ states[i].step_sizes.ω
        @test state_i.halpern.restarts == states[i].halpern.restarts
        @test state_i.halpern.inner == states[i].halpern.inner
    end
    # restarts are decided per instance, so the instances need not restart together
    @test state_batch.halpern.restarts isa Vector{Float64}
end

@testset "Identical batch matches single solve" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    nbatch = 2
    rep(v) = repeat(v, 1, nbatch)
    milp_id = MILP(;
        c = rep(milp.c), lv = rep(milp.lv), uv = rep(milp.uv), milp.A,
        lc = rep(milp.lc), uc = rep(milp.uc), milp.int_var,
    )
    algo = HPDLP(; termination_reltol = 1.0e-6)
    sol, stats = solve(milp_id, algo)
    sol_single, stats_single = solve(milp, algo)
    @test termination_status(stats) == termination_status(stats_single) == MOI.OPTIMAL
    @test stats.kkt_passes == stats_single.kkt_passes
    for i in 1:nbatch
        @test sol.x[:, i] ≈ sol_single.x
    end
end

@testset "Bound and objective rescaling" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    algo = HPDLP(; termination_reltol = 1.0e-6, bound_objective_rescaling = true)
    milp_p, sol_p = preprocess(milp, PrimalDualSolution(milp), algo)
    state = initialize(milp_p, sol_p, algo; starting_time = time())
    # the rescaled problem starts from a unit primal weight
    @test state.step_sizes.ω == 1
    sol, stats = solve(milp, algo)
    @test termination_status(stats) == MOI.OPTIMAL
    sol_ref, _ = solve(milp, PDLP(; termination_reltol = 1.0e-6))
    @test objective_value(sol.x, milp) ≈ objective_value(sol_ref.x, milp) rtol = 1.0e-4
end
