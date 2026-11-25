using CoolPDLP
using CoolPDLP: initialize, instance, restart!, step!, termination_status, hprlp_sigma,
    hprlp_power_iteration
using JLArrays
using KernelAbstractions
using LinearAlgebra
using MathOptBenchmarkInstances
using JuMP: MOI
using Random
using SparseArrays
using Test

@testset "Defaults" begin
    algo = HPRLP()
    @test algo isa CoolPDLP.Algorithm{:HPRLP}
    @test algo.restart.sufficient_decay == 0.2
    @test algo.restart.necessary_decay == 0.6
    @test algo.restart.artificial_decay == 0.2
    @test algo.generic.check_every == 150
    @test algo.preconditioning.curtis_reid_iter == 20
    @test algo.preconditioning.ruiz_iter == 10
    @test algo.preconditioning.chambolle_pock_alpha == 1.0
    @test algo.preconditioning.bound_objective_rescaling
    @test occursin("HPRLP algorithm", sprint(show, algo))
    milp, _ = CoolPDLP.random_milp_and_sol(Xoshiro(0), 5, 8, 0.5)
    @test_throws ArgumentError solve(milp, HPRLP(; check_every = 1))
end

@testset "Power iteration" begin
    A = sprandn(Xoshiro(2), 30, 40, 0.3)
    λ = hprlp_power_iteration(A, sparse(A'))
    @test λ ≈ opnorm(Matrix(A))^2 rtol = 1.0e-3
    # stopped before the first check, it returns the initial estimate
    @test hprlp_power_iteration(A, sparse(A'); maxiter = 3) == 1
end

@testset "Sigma update" begin
    # no restart: unchanged
    @test hprlp_sigma(false, 1.0, 1.0, 4.0, 0.3, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0) == 0.3
    # degenerate movement: back to one
    @test hprlp_sigma(true, 0.0, 1.0, 4.0, 0.3, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0) == 1.0
    # best residual so far: weight exp(-0.05) on the ratio Δx / (Δy √λ) = 2
    f = exp(-0.05)
    @test hprlp_sigma(true, 4.0, 1.0, 4.0, 0.3, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0) ≈ exp(f * log(2) + (1 - f) * log(0.5))
    # near convergence, κ = rel_dual / rel_primal
    @test hprlp_sigma(true, 4.0, 1.0, 4.0, 0.3, 1.0e-12, 1.0e-12, 2.0, 1.0e-12, 4.0e-12, 1.0e-12) ≈ 4 * 2
    @test hprlp_sigma(true, 4.0f0, 1.0f0, 4.0f0, 0.3f0, 1.0f0, 1.0f0, 0.5f0, 1.0f0, 1.0f0, 1.0f0) isa Float32
end

@testset "Convergence on Netlib" begin
    names = ["afiro", "adlittle", "blend", "sc50a"]
    configs = [(SparseMatrixCSC, CPU()), (CoolPDLP.GPUSparseMatrixCSR, JLBackend())]
    @testset "$name on $M in $T" for name in names, (M, backend) in configs, T in (Float64, Float32)
        milp = MILP(read_instance(Netlib, name)[1]; dataset = Netlib, name)
        # in `Float32`, blend has a round-off floor near `1e-3`: the penalty rule drives the
        # primal step `σ = η / ω` up to ~500 there, and `x - σ (c - Aᵀy)` then carries an error
        # `σ ε (|c| + |Aᵀ||y|)` of ~3e-5 per entry and step, which no longer averages out
        # (started at the `Float64` optimum, the iterates drift to a primal error of ~1e-3).
        # `1e-4` is reached or not depending on the summation order of the products.
        blend32 = T == Float32 && name == "blend"
        tol = T == Float64 ? 1.0e-6 : (blend32 ? 1.0e-3 : 1.0e-4)
        algo = HPRLP(T, Int, M; backend, termination_reltol = tol, max_kkt_passes = 10^6)
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        # 1e-4 relative errors in `Float32` leave absolute violations above 1e-2 on afiro
        @test is_feasible(Array(sol.x), milp; cons_tol = T == Float64 ? 1.0e-2 : 1.0e-1)
        sol_ref, _ = solve(milp, PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6))
        @test objective_value(Array(sol.x), milp) ≈ objective_value(sol_ref.x, milp) rtol = (blend32 ? 5.0e-3 : 1.0e-3)
    end
end

@testset "Restart behavior" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    algo = HPRLP(; check_every = 20, termination_reltol = 0.0, max_kkt_passes = 4000)
    milp_p, sol_p = preprocess(milp, PrimalDualSolution(milp), algo)
    state = initialize(milp_p, sol_p, algo; starting_time = time())
    # the first check restarts whatever the residual
    @test !state.halpern.fpe_pending
    @test state.halpern.fpe_initial == Inf
    solve!(state, milp_p, algo)
    @test termination_status(state.stats) == MOI.ITERATION_LIMIT
    @test 1 < state.halpern.restarts < state.stats.kkt_passes / algo.generic.check_every
    @test isfinite(state.best_fpe)
    @test all(isfinite, state.sol.x)
    # the steps keep the PDHG form with η = 1/√λ
    @test state.step_sizes.η ≈ inv(sqrt(state.λ))
end

@testset "Batch matches per-instance solves" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    scales = [1.0, 1.3, 0.7]
    nbatch = length(scales)
    rep(v) = repeat(v, 1, nbatch)
    milp_batch = MILP(;
        c = stack(scale * milp.c for scale in scales), lv = rep(milp.lv), uv = rep(milp.uv),
        milp.A, lc = rep(milp.lc), uc = rep(milp.uc), milp.int_var,
    )
    milps = [instance(milp_batch, i) for i in 1:nbatch]
    algo = HPRLP(; check_every = 10, termination_reltol = 0.0, max_kkt_passes = 400)
    sol_batch = PrimalDualSolution(milp_batch)
    state_batch = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
    solve!(state_batch, milp_batch, algo)
    for i in 1:nbatch
        state = initialize(milps[i], PrimalDualSolution(milps[i]), algo; starting_time = time())
        solve!(state, milps[i], algo)
        state_i = instance(state_batch, i)
        @test state_i.stats.kkt_passes == state.stats.kkt_passes
        @test state_i.sol ≈ state.sol
        @test state_i.step_sizes.ω ≈ state.step_sizes.ω
        @test state_i.halpern.restarts == state.halpern.restarts
    end
end

@testset "Infeasibility detection" begin
    infeasible = MILP(;
        c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 1.0; 1.0 1.0]), lc = [3.0, -Inf], uc = [Inf, 1.0],
    )
    unbounded = MILP(;
        c = [-1.0, 0.0], lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 -1.0]), lc = [-Inf], uc = [1.0],
    )
    _, stats = solve(infeasible, HPRLP(; max_kkt_passes = 10^5))
    @test termination_status(stats) == MOI.INFEASIBLE
    _, stats = solve(unbounded, HPRLP(; max_kkt_passes = 10^5))
    @test termination_status(stats) == MOI.DUAL_INFEASIBLE
    _, stats = solve(infeasible, HPRLP(; max_kkt_passes = 3000, infeasibility_tol = nothing))
    @test termination_status(stats) == MOI.ITERATION_LIMIT
    @test isnothing(stats.ray)
end

@testset "HPRLP σ stays finite on an exactly converged instance" begin
    # `fpe = best_fpe = 0` used to give `exp(-0.05 * 0 / 0) = NaN`
    σ = CoolPDLP.hprlp_sigma(true, 1.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.7, 1.0e-3, 1.0e-3, 1.0e-3)
    @test σ ≈ 0.5
end
