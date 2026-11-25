using CoolPDLP
using CoolPDLP: GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL, KKTErrors,
    Scratch, ZeroMatrix, isquadratic, kkt_errors!, preprocess, relative, termination_status
using HiGHS: HiGHS
using JLArrays: JLBackend
using JuMP: JuMP, MOI
using KernelAbstractions: CPU
using LinearAlgebra
using MathOptBenchmarkInstances
using Random
using SparseArrays
using StableRNGs
using Test

"""
    highs_solution(milp)

Solve the QP `milp` with HiGHS, returning the primal solution and the optimal value.
"""
function highs_solution(milp::MILP)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    n = nbvar(milp)
    JuMP.@variable(model, milp.lv[j] <= x[j = 1:n] <= milp.uv[j])
    JuMP.@constraint(model, milp.lc .<= milp.A * x .<= milp.uc)
    JuMP.@objective(model, Min, x' * milp.Q * x / 2 + milp.c' * x + milp.c0)
    JuMP.optimize!(model)
    @assert JuMP.termination_status(model) == MOI.OPTIMAL
    return JuMP.value.(x), JuMP.objective_value(model)
end

algname(algo::CoolPDLP.Algorithm{A}) where {A} = algo.specific == CoolPDLP.PDLPParameters(true) ? "$A with momentum" : "$A"

qp_algorithms(; kwargs...) = (
    PDHG(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
    PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
    PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, quadratic_momentum = true, kwargs...),
    HPDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
)

@testset "Linear programs keep a ZeroMatrix" begin
    milp = CoolPDLP.random_milp_and_sol(StableRNG(0), 5, 4, 0.5)[1]
    @test milp.Q isa ZeroMatrix{Float64}
    @test !isquadratic(milp.Q)
    @test size(milp.Q) == (4, 4)
    @test iszero(Matrix(milp.Q))
    milp_p, _ = preprocess(milp, PrimalDualSolution(milp), PDLP(Float32, Int32, GPUSparseMatrixCSR))
    @test milp_p.Q isa ZeroMatrix{Float32}
end

@testset "Linear programs read from a file stay linear" begin
    qps, _ = read_instance(Netlib, "afiro")
    milp = MILP(qps)
    @testset "$M" for M in (Matrix, SparseMatrixCSC, GPUSparseMatrixCSR, GPUSparseMatrixELL)
        milp_p, _ = preprocess(milp, PrimalDualSolution(milp), PDLP(Float64, Int32, M))
        @test !isquadratic(milp_p.Q)
    end
    # a quadratic one keeps its products by `Q` with dense matrices
    qp = MILP(read_instance(MarosMeszaros, "HS21")[1])
    qp_p, _ = preprocess(qp, PrimalDualSolution(qp), PDLP(Float64, Int32, Matrix))
    @test isquadratic(qp_p.Q)
    @test Matrix(qp_p.Q) ≈ Matrix(preprocess(qp, PrimalDualSolution(qp), PDLP(Float64, Int32, SparseMatrixCSC))[1].Q)
end

@testset "Batch-scaled Q in Float32" begin
    # bound and objective rescaling gives every instance its own scale of `Q`, which must follow
    # the element type of the algorithm
    rng = StableRNG(1)
    Q = sparse([2.0 1.0; 1.0 2.0])
    qp = MILP(;
        c = randn(rng, 2, 3), Q, lv = zeros(2), uv = fill(10.0, 2),
        A = sparse([1.0 1.0]), lc = [-Inf], uc = [1.0],
    )
    algo = PDLP(Float32, Int32, GPUSparseMatrixCSR; bound_objective_rescaling = true)
    milp_p, _ = preprocess(qp, PrimalDualSolution(qp), algo)
    @test milp_p.Q isa CoolPDLP.BatchScaledMatrix
    @test eltype(milp_p.Q.scale) == Float32
    @test eltype(CoolPDLP.set_eltype(Float32, CoolPDLP.BatchScaledMatrix(Q, [1.0, 2.0])).scale) == Float32
    # one instance of it, in a sparse format of this package
    Q1 = CoolPDLP.instance_quadratic(milp_p.Q, 2)
    @test Q1 isa GPUSparseMatrixCSR
    @test Matrix(Q1) ≈ milp_p.Q.scale[2] * Matrix(milp_p.Q.Q)
end

@testset "Hand-made QPs" begin
    # min ½(x₁² + x₂²) - x₁ - x₂  s.t.  x₁ + x₂ ≤ 1, x ≥ 0: x = (½, ½), y = -½
    simplex = MILP(;
        c = [-1.0, -1.0], Q = sparse(1.0I, 2, 2), lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 1.0]), lc = [-Inf], uc = [1.0],
    )
    # min x₁² + x₁x₂ + x₂² + x₃² - 3 x₃  s.t.  x₁ + x₂ = 2, 0 ≤ x₃ ≤ 1: x = (1, 1, 1), y = 3
    coupled = MILP(;
        c = [0.0, 0.0, -3.0], Q = sparse([2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 2.0]),
        lv = [-Inf, -Inf, 0.0], uv = [Inf, Inf, 1.0],
        A = sparse([1.0 1.0 0.0]), lc = [2.0], uc = [2.0], c0 = 1.0,
    )
    # no constraints at all, only a box: x = (1, -2) clamped to [-1, 1]² is (1, -1)
    box = MILP(;
        c = [-2.0, 4.0], Q = sparse(2.0I, 2, 2), lv = [-1.0, -1.0], uv = [1.0, 1.0],
        A = spzeros(0, 2), lc = Float64[], uc = Float64[],
    )
    cases = [
        (simplex, [0.5, 0.5], [-0.5], -0.75),
        (coupled, [1.0, 1.0, 1.0], [3.0], 3.0 + 1.0 - 3.0 + 1.0),
        (box, [1.0, -1.0], Float64[], 1.0 - 2.0 + 1.0 - 4.0),
    ]
    @testset "$(algname(algo))" for algo in qp_algorithms()
        for (milp, x, y, obj) in cases
            sol, stats = solve(milp, algo)
            @test termination_status(stats) == MOI.OPTIMAL
            @test sol.x ≈ x atol = 1.0e-4
            @test sol.y ≈ y atol = 1.0e-4
            @test objective_value(sol.x, milp) ≈ obj atol = 1.0e-4
        end
    end
    @testset "KKT errors vanish at the optimum" begin
        for (milp, x, y, _) in cases
            sol = PrimalDualSolution(x, y)
            err = kkt_errors!(KKTErrors(sol), Scratch(sol), sol, milp, L2Norm())
            @test relative(err) < 1.0e-12
        end
    end
end

@testset "Random convex QPs against HiGHS" begin
    rng = StableRNG(42)
    instances = [
        random_qp(rng, 10, 15),
        random_qp(rng, 20, 30; density = 0.2),
        random_qp(rng, 15, 25; rank = 5),
    ]
    @testset "instance $k" for (k, milp) in enumerate(instances)
        x_ref, obj_ref = highs_solution(milp)
        @testset "$(algname(algo))" for algo in qp_algorithms()
            sol, stats = solve(milp, algo)
            @test termination_status(stats) == MOI.OPTIMAL
            @test is_feasible(sol.x, milp; cons_tol = 1.0e-4, verbose = false)
            @test objective_value(sol.x, milp) ≈ obj_ref rtol = 1.0e-4
        end
    end
end

@testset "Matrix formats and Float32" begin
    milp = random_qp(StableRNG(1), 10, 15)
    _, obj_ref = highs_solution(milp)
    configs = [
        (Float64, SparseMatrixCSC, CPU()),
        (Float32, SparseMatrixCSC, CPU()),
        (Float64, GPUSparseMatrixCSR, JLBackend()),
        (Float64, GPUSparseMatrixCOO, JLBackend()),
        (Float32, GPUSparseMatrixELL, JLBackend()),
    ]
    @testset "$T, $M" for (T, M, backend) in configs
        tol = T == Float32 ? 1.0e-3 : 1.0e-6
        for algo in (
                PDLP(T, Int, M; backend, termination_reltol = tol),
                HPDLP(T, Int, M; backend, termination_reltol = tol),
            )
            sol, stats = solve(milp, algo)
            @test termination_status(stats) == MOI.OPTIMAL
            @test eltype(sol.x) == T
            @test objective_value(Array{Float64}(Array(sol.x)), milp) ≈ obj_ref rtol = 10tol
        end
    end
end

@testset "Batched QPs" begin
    rng = StableRNG(3)
    milp = random_qp(rng, 10, 15)
    nbatch = 3
    cs = [milp.c .+ 0.3 .* randn(rng, length(milp.c)) for _ in 1:nbatch]
    singles = [MILP(; c, milp.c0, milp.Q, milp.lv, milp.uv, milp.A, milp.lc, milp.uc) for c in cs]
    batch = MILP(; c = stack(cs), milp.c0, milp.Q, milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    @test isbatched(batch)
    @testset "$(algname(algo))" for algo in qp_algorithms()
        sol, stats = solve(batch, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        objs = objective_value(sol.x, batch)
        for i in 1:nbatch
            _, obj_ref = highs_solution(singles[i])
            @test objs[i] ≈ obj_ref rtol = 1.0e-4
        end
    end
end

@testset "Batched QPs with per-instance bound and objective rescaling" begin
    rng = StableRNG(4)
    milp = random_qp(rng, 10, 15)
    # objectives of very different sizes, so that each instance needs its own ratio γ / β
    cs = [f .* milp.c for f in (1.0, 30.0, 0.05)]
    singles = [MILP(; c, milp.c0, milp.Q, milp.lv, milp.uv, milp.A, milp.lc, milp.uc) for c in cs]
    batch = MILP(; c = stack(cs), milp.c0, milp.Q, milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    milp_scaled, _ = preprocess(batch, PrimalDualSolution(batch), PDLP(; bound_objective_rescaling = true))
    S = milp_scaled.Q
    @test S isa CoolPDLP.BatchScaledMatrix
    @test !allequal(S.scale)
    # the products by `S` are those by each instance's own matrix
    X = randn(rng, nbvar(batch), 3)
    QX = CoolPDLP.quadratic_mul!(similar(X), S, X)
    for i in 1:3
        @test QX[:, i] ≈ (S.scale[i] * S.Q) * X[:, i]
        @test instance(milp_scaled, i).Q ≈ S.scale[i] * S.Q
    end
    G = randn(rng, nbvar(batch), 3)
    @test CoolPDLP.subtract_quadratic_gradient!(copy(G), S, X) ≈ G - QX
    configs = ((SparseMatrixCSC, CPU()), (GPUSparseMatrixCSR, JLBackend()))
    @testset "$(algname(algo)) on $M" for (M, backend) in configs, algo in (
                PDLP(Float64, Int, M; backend, termination_reltol = 1.0e-6, max_kkt_passes = 10^6, bound_objective_rescaling = true),
                HPDLP(Float64, Int, M; backend, termination_reltol = 1.0e-6, max_kkt_passes = 10^6, bound_objective_rescaling = true),
            )
        sol, stats = solve(batch, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        x = Array(sol.x)
        for i in 1:3
            _, obj_ref = highs_solution(singles[i])
            @test objective_value(x[:, i], singles[i]) ≈ obj_ref rtol = 1.0e-4
        end
    end
end

@testset "PDQP momentum changes the iterates" begin
    qp = MILP(read_instance(MarosMeszaros, "HS118")[1])
    base = (; termination_reltol = 1.0e-8, max_kkt_passes = 200, check_every = 1000)
    sol0, stats0 = solve(qp, PDLP(; base...))
    sol1, stats1 = solve(qp, PDLP(; base..., quadratic_momentum = true))
    @test stats1.kkt_passes == stats0.kkt_passes
    @test sol1.x != sol0.x
    # the aggregate iterate is that of PDQP, not the η-weighted average of PDLP
    milp, sol = preprocess(qp, PrimalDualSolution(qp), PDLP(; base..., quadratic_momentum = true))
    state = CoolPDLP.initialize(milp, sol, PDLP(; quadratic_momentum = true); starting_time = 0.0)
    algo = PDLP(; quadratic_momentum = true)
    CoolPDLP.step!(state, milp, algo)
    @test state.sol_avg.x == state.sol.x  # β = 1 at the first step
    x_avg = copy(state.sol_avg.x)
    CoolPDLP.step!(state, milp, algo)
    @test state.sol_avg.x ≈ (2 / 3) * state.sol.x + (1 / 3) * x_avg
end

@testset "Maros-Mészáros" begin
    # small instances which every algorithm solves within the budget below
    names = ["HS21", "HS35", "HS118", "QAFIRO", "GENHS28"]
    known = Dict(
        "HS21" => -9.99599999e+1, "HS35" => 1.11111111e-1, "HS118" => 6.64820452e+2,
        "QAFIRO" => -1.59078179e+0, "GENHS28" => 9.27173692e-1,
    )
    @testset "$name" for name in names
        qps, path = read_instance(MarosMeszaros, name)
        @test_throws ArgumentError MILP(qps, Val(false))
        milp = MILP(qps; dataset = MarosMeszaros, path)
        @test isquadratic(milp.Q)
        @testset "$(algname(algo))" for algo in qp_algorithms(; termination_reltol = 1.0e-6, time_limit = 60.0)
            sol, stats = solve(milp, algo)
            @test termination_status(stats) == MOI.OPTIMAL
            @test objective_value(sol.x, milp) ≈ known[name] rtol = 1.0e-4
        end
    end
end


@testset "Algorithms without quadratic support" begin
    @test all(CoolPDLP.supports_quadratic, (PDHG(), PDLP(), HPDLP()))
    # any algorithm that does not declare the support refuses a quadratic program up front
    algo = CoolPDLP.Algorithm{:NoQuadratic}(Float64, Int, SparseMatrixCSC)
    @test !CoolPDLP.supports_quadratic(algo)
    qp = MILP(; c = [1.0], Q = sparse([1.0;;]), lv = [0.0], uv = [1.0], A = sparse([1.0;;]), lc = [0.0], uc = [1.0])
    @test_throws ArgumentError solve(qp, algo)
end

@testset "Quadratic helpers" begin
    lp = MILP(read_instance(Netlib, "afiro")[1], Val(false))
    @test lp.Q isa ZeroMatrix{Float64}
    qp = MILP(read_instance(MarosMeszaros, "HS21")[1])
    @test occursin("quadratic objective nonzeros", sprint(show, qp))
    Q = sparse([2.0 0; 0 4])
    x = [1.0, 1.0]
    @test CoolPDLP.mul_quadratic!(zeros(2), Q, x) == [2.0, 4.0]
    @test CoolPDLP.mul_quadratic!(ones(2), spzeros(2, 2), x) == zeros(2)
    @test CoolPDLP.scaled_quadratic_term([6.0, 8.0], [1.0, 2.0], [3.0, 2.0]) ≈ [2.0, 2.0]
    # a second rescaling of a batch-scaled matrix multiplies the ratios
    S = CoolPDLP.rescale_quadratic(Q, [1.0, 2.0], [2.0, 2.0])
    @test S isa CoolPDLP.BatchScaledMatrix
    S2 = CoolPDLP.rescale_quadratic(S, [1.0, 2.0], [2.0, 2.0])
    @test S2.Q === Q
    @test S2.scale ≈ [4.0, 1.0]
end
