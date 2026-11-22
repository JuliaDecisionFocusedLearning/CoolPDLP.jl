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

algname(algo::CoolPDLP.Algorithm{A}) where {A} = "$A"

qp_algorithms(; kwargs...) = (
    PDHG(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
    PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
    HPDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
    HPRQP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, kwargs...),
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
            err = kkt_errors!(KKTErrors(sol), Scratch(sol), sol, milp)
            @test relative(err) < 1.0e-12
        end
    end
end

@testset "PDQP relative KKT error" begin
    milp = random_qp(StableRNG(7), 10, 15)
    sol = PrimalDualSolution(randn(StableRNG(8), 15), randn(StableRNG(9), 10))
    err = kkt_errors!(KKTErrors(sol), Scratch(sol), sol, milp)
    err_pdqp = kkt_errors!(KKTErrors(sol), Scratch(sol), sol, milp, Val(:pdqp))
    (; x, y) = sol
    (; c, Q, A, lv, uv, lc, uc) = milp
    Ax = A * x
    b = CoolPDLP.combine.(lc, uc)
    r = c + Q * x - A' * y
    z = CoolPDLP.proj_multiplier.(r, lv, uv)
    @test err_pdqp.primal ≈ norm(Ax - clamp.(Ax, lc, uc), Inf)
    @test err_pdqp.primal_scale ≈ 1 + max(norm(b, Inf), norm(Ax, Inf))
    @test err_pdqp.dual ≈ norm(r - z, Inf)
    @test err_pdqp.dual_scale ≈ 1 + max(norm(c, Inf), norm(Q * x, Inf), norm(A' * y, Inf))
    @test err_pdqp.gap ≈ err.gap
    @test err_pdqp.gap_scale <= err.gap_scale
    sol_opt, stats = solve(milp, PDLP(; relative_kkt = Val(:pdqp), termination_reltol = 1.0e-6))
    @test termination_status(stats) == MOI.OPTIMAL
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
                HPRQP(T, Int, M; backend, termination_reltol = tol),
            )
            sol, stats = solve(milp, algo)
            @test termination_status(stats) == MOI.OPTIMAL
            @test eltype(sol.x) == T
            @test objective_value(Array{Float64}(Array(sol.x)), milp) ≈ obj_ref rtol = 10tol
        end
    end
end

@testset "HPRQP handles a diagonal Q exactly on every format" begin
    milp = random_qp(StableRNG(5), 10, 15)
    diag_qp = MILP(; milp.c, Q = sparse(Diagonal(diag(milp.Q))), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    x = zeros(15)
    λ = 7.0
    @test CoolPDLP.proximal_diagonal(diag_qp.Q, λ, x) == diag(milp.Q)
    @test CoolPDLP.proximal_diagonal(milp.Q, λ, x) == fill(λ, 15)
    for M in (GPUSparseMatrixCSR, GPUSparseMatrixCOO, GPUSparseMatrixELL)
        Q_M = M(diag_qp.Q)
        @test Array(CoolPDLP.proximal_diagonal(Q_M, λ, x)) == diag(milp.Q)
        @test Array(CoolPDLP.proximal_diagonal(M(milp.Q), λ, x)) == fill(λ, 15)
    end
    # the same iterates as with `SparseMatrixCSC`, hence the same number of passes
    _, stats_csc = solve(diag_qp, HPRQP(; termination_reltol = 1.0e-6))
    _, stats_csr = solve(diag_qp, HPRQP(Float64, Int, GPUSparseMatrixCSR; backend = JLBackend(), termination_reltol = 1.0e-6))
    @test termination_status(stats_csc) == termination_status(stats_csr) == MOI.OPTIMAL
    @test stats_csc.kkt_passes == stats_csr.kkt_passes
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
            HPRQP(Float64, Int, M; backend, termination_reltol = 1.0e-6, max_kkt_passes = 10^6, bound_objective_rescaling = true),
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

@testset "HPRQP follows HPR-QP on DUAL2" begin
    # without HPR-QP's symmetric Gauss-Seidel correction of the dual step, the iteration
    # oscillated on DUAL2 instead of converging in about 600 steps like HPR-QP.jl does
    milp = MILP(read_instance(MarosMeszaros, "DUAL2")[1])
    algo = HPRQP(; termination_reltol = 1.0e-6, relative_kkt = Val(:pdqp), max_kkt_passes = 2000)
    sol, stats = solve(milp, algo)
    @test termination_status(stats) == MOI.OPTIMAL
    @test stats.kkt_passes <= 1000
    @test objective_value(sol.x, milp) ≈ 3.37336761e-2 rtol = 1.0e-5
end

@testset "HPRQP σ models against a brute-force minimum" begin
    grid = exp.(range(log(1.0e-6), log(1.0e6), length = 400_001))
    brute(f) = grid[argmin(f.(grid))]
    # scalar model (general `Q`)
    a, b, c, d = 3.0, 0.5, 2.0, 4.0
    @test CoolPDLP.golden_sigma(a, b, c, d) ≈ brute(s -> a * s + b / s + c * s^2 / (1 + d * s)) rtol = 1.0e-3
    # per-coordinate model (diagonal `Q`), on the state of a diagonal QP, batched and not
    rng = StableRNG(0)
    n, m = 6, 3
    Q = sparse(Diagonal(rand(rng, n) .+ 0.1))
    A = sprandn(rng, m, n, 0.8) + sparse(I, m, n)
    milp = MILP(; c = randn(rng, n), lv = fill(-1.0, n), uv = fill(1.0, n), A, lc = fill(-1.0, m), uc = fill(1.0, m), Q)
    algo = HPRQP()
    pmilp, sol = preprocess(milp, PrimalDualSolution(milp), algo)
    state = CoolPDLP.initialize(pmilp, sol, algo; starting_time = 0.0)
    @test state.Q_diagonal
    aty = randn(rng, n)
    state.buf3 .= aty
    CoolPDLP.mul_quadratic!(state.scratch.z, pmilp.Q, state.buf3)
    qaty = copy(state.scratch.z)
    state.t1, state.t2 = 3.0, 0.5
    σ = CoolPDLP.golden_sigma_diagonal!!(state)
    f(s) = 3.0 * s + 0.5 / s + s^2 * sum(aty .* qaty ./ (1 .+ s .* state.λ_Qv))
    @test σ ≈ brute(f) rtol = 1.0e-3
    # a general `Q` does not take the per-coordinate model
    milp_general = MILP(; c = milp.c, lv = milp.lv, uv = milp.uv, A, lc = milp.lc, uc = milp.uc, Q = Q + sparse([1, 2], [2, 1], [0.01, 0.01], n, n))
    pg, sg = preprocess(milp_general, PrimalDualSolution(milp_general), algo)
    @test !CoolPDLP.initialize(pg, sg, algo; starting_time = 0.0).Q_diagonal
end
