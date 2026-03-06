using CoolPDLP
using Random
using JLArrays
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Test

# min (1/2)(x₁² + x₂²) s.t. x₁ + x₂ = 1, x ≥ 0
function simple_equality_qp()
    n = 2
    c = zeros(n)
    Q = sparse(1.0I, n, n)  # Identity Hessian
    A = sparse([1.0 1.0])
    lv = zeros(n)
    uv = fill(Inf, n)
    lc = [1.0]
    uc = [1.0]
    return QuadraticProgram(; c, Q, lv, uv, A, lc, uc)
end

# min (1/2)(2x₁² + x₂²) + x₁ + 2x₂ s.t. x₁ + x₂ ≥ 1, x ≥ 0
function linear_quadratic_qp()
    n = 2
    c = [1.0, 2.0]
    Q = sparse([2.0 0.0; 0.0 1.0])
    A = sparse([1.0 1.0])
    lv = zeros(n)
    uv = fill(Inf, n)
    lc = [1.0]
    uc = [Inf]
    return QuadraticProgram(; c, Q, lv, uv, A, lc, uc)
end

function random_qp(n, m; seed = 42)
    rng = Random.MersenneTwister(seed)
    B = randn(rng, n, n)
    Q_dense = B' * B + 0.1I
    Q = sparse(Q_dense)
    c = randn(rng, n)
    A = sprandn(rng, m, n, 0.3)
    x_feas = abs.(randn(rng, n))
    b = A * x_feas
    lv = zeros(n)
    uv = fill(Inf, n)
    lc = b
    uc = b
    return QuadraticProgram(; c, Q, lv, uv, A, lc, uc)
end

@testset "QP - PDLP" begin
    @testset "Simple equality QP" begin
        milp = simple_equality_qp()
        @test milp isa QuadraticProgram
        @test milp isa QuadraticProgram
        algo = PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test sol.x[1] ≈ 1 / 2 atol = 1.0e-4
        @test sol.x[2] ≈ 1 / 2 atol = 1.0e-4
        @test objective_value(sol.x, milp) ≈ 0.25 atol = 1.0e-4
    end

    @testset "Linear + quadratic QP" begin
        milp = linear_quadratic_qp()
        algo = PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test sol.x[1] ≈ 2 / 3 atol = 1.0e-3
        @test sol.x[2] ≈ 1 / 3 atol = 1.0e-3
    end

    @testset "Random QP" begin
        milp = random_qp(20, 5)
        algo = PDLP(; termination_reltol = 1.0e-4, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test is_feasible(sol.x, milp; cons_tol = 1.0e-2)
    end
end
