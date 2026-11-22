using CoolPDLP
using CoolPDLP: HSDPoint, boxcone_root!, hsd_operator!, initialize, instance, termination_status
using GPUArraysCore: @allowscalar
using JLArrays
using KernelAbstractions
using LinearAlgebra
using MathOptBenchmarkInstances
using JuMP: MOI
using Random
using SparseArrays
using Test

@testset "Defaults follow the paper" begin
    algo = HSDPDLP()
    @test algo isa CoolPDLP.Algorithm{:HSDPDLP}
    @test algo.step_size.invnorm_scaling == 0.998
    @test algo.step_size.pid_kp == 0.97
    @test algo.step_size.pid_ki == 0.05
    @test algo.step_size.pid_kd == 0.0
    @test algo.step_size.pid_integral_discount == 0.3
    @test algo.restart.sufficient_decay == 0.2
    @test algo.restart.necessary_decay == 0.5
    @test algo.restart.artificial_decay == 0.36
    @test algo.generic.check_every == 200
    @test occursin("HSDPDLP algorithm", sprint(show, algo))
    milp = CoolPDLP.random_milp_and_sol(Xoshiro(0), 5, 8, 0.5)[1]
    @test_throws ArgumentError solve(milp, HSDPDLP(; check_every = 1))
end

"""
    random_bounds(rng, k)

Random bounds mixing every kind of interval: free, one-sided, two-sided and equalities.
"""
function random_bounds(rng, k)
    l, u = randn(rng, k), randn(rng, k)
    l, u = min.(l, u), max.(l, u)
    kind = rand(rng, 1:5, k)
    l[kind .== 1] .= -Inf
    u[kind .== 1] .= Inf
    l[kind .== 2] .= -Inf
    u[kind .== 3] .= Inf
    u[kind .== 4] .= l[kind .== 4]
    return l, u
end

@testset "Box-cone projection" begin
    # the Euclidean projection onto K[l, u] minimizes the strongly convex φ(τ) of eq. 31, which
    # a ternary search on a wide bracket finds independently of the Newton method
    rng = Xoshiro(0)
    m = 30
    lc, uc = random_bounds(rng, m)
    milp = MILP(; c = randn(rng, 4), lv = fill(-Inf, 4), uv = fill(Inf, 4), A = sprandn(rng, m, 4, 0.5), lc, uc)
    state = initialize(milp, PrimalDualSolution(milp), HSDPDLP(); starting_time = time())
    ws = state.ws
    clip(w, l, u, τ) = CoolPDLP.box_clip(w, l, u, τ)
    φ(τ, t̂, ŵ) = (τ - t̂)^2 / 2 + sum(abs2, clip.(ŵ, lc, uc, τ) .- ŵ) / 2
    for trial in 1:50
        t̂ = 3 * randn(rng)
        ŵ = 3 * randn(rng, m)
        ws.wt .= ŵ
        ws.tt = t̂
        warm = trial % 2 == 0 ? 0.0 : abs(randn(rng))
        τ = boxcone_root!(ws, (lc, uc), warm, 1.0)
        a, b = 0.0, 100.0
        for _ in 1:200
            m1, m2 = a + (b - a) / 3, b - (b - a) / 3
            φ(m1, t̂, ŵ) < φ(m2, t̂, ŵ) ? (b = m2) : (a = m1)
        end
        @test τ >= 0
        @test τ ≈ (a + b) / 2 atol = 1.0e-5 rtol = 1.0e-5
    end
end

"""
    P_inner(a, b, A, μ, η, σ)

`⟨a, P b⟩` in the metric `P` of the HSD operator (eq. 24).
"""
function P_inner(a::HSDPoint, b::HSDPoint, A, μ, η, σ)
    return a.τ * (b.τ / μ - b.α) + dot(a.xy.x, b.xy.x / η - A' * b.xy.y) +
        a.α * (b.α / σ - b.τ) + dot(a.xy.y, b.xy.y / σ - A * b.xy.x)
end

randpoint(rng, n, m) = HSDPoint(PrimalDualSolution(randn(rng, n), randn(rng, m)), abs(randn(rng)), randn(rng))
diffpoint(a, b) = HSDPoint(PrimalDualSolution(a.xy.x - b.xy.x, a.xy.y - b.xy.y), a.τ - b.τ, a.α - b.α)

@testset "Matrix-product and Newton accounting" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    algo = HSDPDLP(; check_every = 50, max_kkt_passes = 500, infeasibility_tol = nothing)
    sol, stats = solve(milp, algo)
    checks = stats.kkt_passes ÷ 50
    # per check of 50 steps: 2 residual products, 2 + 3 for the two optimality candidates, 1 for
    # the certificate regime; plus 2 for the initial KKT errors
    @test CoolPDLP.matrix_products(stats) == 2 * stats.kkt_passes + 8 * checks + 2
    @test stats.newton_iterations >= 2 * stats.kkt_passes
end

@testset "Active-set reduced cost" begin
    arc = CoolPDLP.active_reduced_cost
    # at its lower bound with a positive reduced cost: kept; inactive or wrong sign: dropped
    @test arc(3.0, 1.0, 0.0, 0.0, Inf, 1.0) == 2.0
    @test arc(3.0, 1.0, 0.5, 0.0, Inf, 1.0) == 0.0
    @test arc(1.0, 3.0, 0.0, 0.0, Inf, 1.0) == 0.0
    @test arc(1.0, 3.0, 2.0, 0.0, 1.0, 2.0) == -2.0
    @test arc(1.0, 3.0, 2.0, -Inf, Inf, 2.0) == 0.0
end

@testset "Newton stops at machine precision" begin
    # grow7: the Newton step underflows (g / d below the ulp of t) while |g| stays above the
    # d_min-based threshold of the paper; the iteration must stop instead of looping
    t, g, d, dmin = 0.6374158698452873, 4.735914617139937e-9, 6.954478284283043e8, 1.0
    t2, lo2, hi2, Δ2, done2 = CoolPDLP.newton_update(t, g, d, 0.0, t, t, 1.0, false, dmin, 1.0e-9)
    @test done2
    @test t2 == t
    # a step which still moves is taken
    t2, _, _, _, done2 = CoolPDLP.newton_update(1.0, 0.5, 1.0, 0.0, 1.0, 1.0, Inf, false, 1.0, 1.0e-9)
    @test !done2
    @test t2 == 0.5
end

@testset "Operator" begin
    rng = Xoshiro(1)
    n, m = 12, 9
    lv, uv = random_bounds(rng, n)
    lc, uc = random_bounds(rng, m)
    milp = MILP(; c = randn(rng, n), lv, uv, A = sprandn(rng, m, n, 0.5), lc, uc)
    algo = HSDPDLP()
    state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
    (; ws, hsd, step_sizes) = state
    step_sizes.ω = 1.7
    ν = algo.step_size.invnorm_scaling
    T(z) = hsd_operator!(copy(z), z, ws, hsd, step_sizes, ν)
    lv, uv, lc, uc = hsd.lv, hsd.uv, hsd.lc, hsd.uc
    for _ in 1:20
        z, w = randpoint(rng, n, m), randpoint(rng, n, m)
        Tz, Tw = T(z), T(w)
        μ, η, σ = ws.μ, ws.ηx, ws.σ
        # the image lies in K = K[lv, uv] × K[lc, uc]*
        @test Tz.τ >= 0
        @test all(@. ifelse(isfinite(lv), Tz.xy.x >= lv * Tz.τ - 1.0e-9, true))
        @test all(@. ifelse(isfinite(uv), Tz.xy.x <= uv * Tz.τ + 1.0e-9, true))
        p = sum(@. CoolPDLP.safeprod_left(uc, CoolPDLP.negative_part(Tz.xy.y)) - CoolPDLP.safeprod_left(lc, CoolPDLP.positive_part(Tz.xy.y)))
        @test Tz.α >= p - 1.0e-9
        # firm nonexpansiveness in the P-norm, which pins down the whole operator
        ΔT, Δ = diffpoint(Tz, Tw), diffpoint(z, w)
        @test P_inner(ΔT, ΔT, hsd.A, μ, η, σ) <= P_inner(ΔT, Δ, hsd.A, μ, η, σ) + 1.0e-6
    end
    # a solution of the embedding is a fixed point: an optimal pair with τ = 1 and α = p(-y)
    feasible = MILP(; c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf], A = sparse([1.0 1.0]), lc = [1.0], uc = [Inf])
    state = initialize(feasible, PrimalDualSolution(feasible), algo; starting_time = time())
    (; ws, hsd, step_sizes) = state
    γb, γc = ws.γb, ws.γc
    zstar = HSDPoint(PrimalDualSolution([γb, 0.0], [γc]), 1.0, -γb * γc)
    Tz = hsd_operator!(copy(zstar), zstar, ws, hsd, step_sizes, ν)
    @test Tz.xy.x ≈ zstar.xy.x atol = 1.0e-9
    @test Tz.xy.y ≈ zstar.xy.y atol = 1.0e-9
    @test Tz.τ ≈ 1 atol = 1.0e-9
    @test Tz.α ≈ zstar.α atol = 1.0e-9
end

@testset "Convergence on Netlib" begin
    names = ["afiro", "adlittle", "blend", "sc50a"]
    configs = [(SparseMatrixCSC, CPU()), (CoolPDLP.GPUSparseMatrixCSR, JLBackend())]
    @testset "$name on $M" for name in names, (M, backend) in configs
        milp = MILP(read_instance(Netlib, name)[1]; dataset = Netlib, name)
        algo = HSDPDLP(Float64, Int, M; backend, termination_reltol = 1.0e-6, max_kkt_passes = 10^6)
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        @test is_feasible(Array(sol.x), milp; cons_tol = 1.0e-3)
        sol_ref, _ = solve(milp, PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6))
        @test objective_value(Array(sol.x), milp) ≈ objective_value(sol_ref.x, milp) rtol = 1.0e-4
    end
end

@testset "Float32 and matrix formats" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    obj = objective_value(solve(milp, PDLP(; termination_reltol = 1.0e-6))[1].x, milp)
    configs = [
        (Float32, SparseMatrixCSC, CPU()), (Float64, Matrix, CPU()),
        (Float64, CoolPDLP.GPUSparseMatrixCSR, JLBackend()),
        (Float64, CoolPDLP.GPUSparseMatrixELL, JLBackend()),
        (Float64, CoolPDLP.GPUSparseMatrixCOO, JLBackend()),
    ]
    @testset "$T with $M" for (T, M, backend) in configs
        algo = HSDPDLP(T, Int, M; backend, termination_reltol = 1.0e-4, max_kkt_passes = 10^5)
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        @test eltype(sol.x) == T
        @test objective_value(Float64.(Array(sol.x)), milp) ≈ obj rtol = 1.0e-2
    end
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
    # an unreachable tolerance, so that every solve runs for the same number of iterations
    algo = HSDPDLP(; check_every = 10, termination_reltol = 0.0, max_kkt_passes = 400)
    state_batch = initialize(milp_batch, PrimalDualSolution(milp_batch), algo; starting_time = time())
    solve!(state_batch, milp_batch, algo)
    states = map(1:nbatch) do i
        state = initialize(milps[i], PrimalDualSolution(milps[i]), algo; starting_time = time())
        solve!(state, milps[i], algo)
    end
    @test allunique(eachcol(state_batch.sol.x))
    for i in 1:nbatch
        state_i = instance(state_batch, i)
        @test state_i.stats.kkt_passes == states[i].stats.kkt_passes
        @test state_i.sol ≈ states[i].sol
        @test state_i.z.xy ≈ states[i].z.xy
        @test state_i.z.τ ≈ states[i].z.τ
        @test state_i.step_sizes.ω ≈ states[i].step_sizes.ω
        @test state_i.halpern.restarts == states[i].halpern.restarts
    end

    # a full solve of the batch reaches the same solutions
    algo = HSDPDLP(; termination_reltol = 1.0e-6)
    sol, stats = solve(milp_batch, algo)
    @test termination_status(stats) == MOI.OPTIMAL
    for i in 1:nbatch
        sol_i, _ = solve(milps[i], algo)
        @test objective_value(sol.x[:, i], milps[i]) ≈ objective_value(sol_i.x, milps[i]) rtol = 1.0e-4
    end
end

@testset "Type stability" begin
    milp = MILP(read_instance(Netlib, "afiro")[1]; dataset = Netlib, name = "afiro")
    algo = HSDPDLP(; record_error_history = false)
    state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
    @inferred CoolPDLP.hsd_step!(state, algo, Val(true))
    @inferred CoolPDLP.hsd_restart!(state, algo)
    @inferred instance(state, 1)
end
