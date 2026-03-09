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

function high_linear_weight_qp()
    n = 2
    c = [100.0, 100.0]
    Q = sparse([10.0 0.0; 0.0 10.0])
    A = sparse([1.0 1.0])
    lv = zeros(n)
    uv = fill(Inf, n)
    lc = [1.0]
    uc = [1.0]
    return QuadraticProgram(; c, Q, lv, uv, A, lc, uc)
end

@testset "QP - PDLP" begin
    @testset "Simple equality QP" begin
        milp = simple_equality_qp()
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

@testset "QP - PDHG" begin
    @testset "Simple equality QP" begin
        milp = simple_equality_qp()
        algo = PDHG(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test sol.x[1] ≈ 1 / 2 atol = 1.0e-4
        @test sol.x[2] ≈ 1 / 2 atol = 1.0e-4
    end

    @testset "Linear + quadratic QP" begin
        milp = linear_quadratic_qp()
        algo = PDHG(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test sol.x[1] ≈ 2 / 3 atol = 1.0e-3
        @test sol.x[2] ≈ 1 / 3 atol = 1.0e-3
    end

    @testset "High linear weight QP" begin
        milp = high_linear_weight_qp()
        algo = PDHG(; termination_reltol = 1.0e-6, max_kkt_passes = 10^6, show_progress = false)
        sol, stats = solve(milp, algo)
        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test sol.x[1] ≈ 1 / 2 atol = 1.0e-4
        @test sol.x[2] ≈ 1 / 2 atol = 1.0e-4
    end
end

@testset "QP step-size numerics" begin
    @testset "compute_eta" begin
        T = Float64
        # norm_A and norm_Q nonzero: η = invnorm_scaling * 2 / (hypot(b, 2*norm_A) + b), b = norm_Q/(2ω)
        let norm_A = 2.0, norm_Q = 4.0, ω = 1.5, s = 0.9
            b = norm_Q / (2ω)
            @test CoolPDLP.compute_eta(norm_A, norm_Q, ω, s) ≈ s * 2 / (hypot(b, 2norm_A) + b)
        end

        # norm_A = 0 (Q-only): η = invnorm_scaling * 2ω / norm_Q
        let norm_Q = 4.0, ω = 2.0, s = 0.5
            @test CoolPDLP.compute_eta(zero(T), norm_Q, ω, s) ≈ s * 2ω / norm_Q
        end

        # norm_Q = 0 (LP): η = invnorm_scaling / norm_A
        let norm_A = 2.0, s = 0.9
            @test CoolPDLP.compute_eta(norm_A, zero(T), 1.0, s) ≈ s / norm_A
        end

        # both zero: fallback to 1.0 regardless of ω and invnorm_scaling
        let
            @test CoolPDLP.compute_eta(zero(T), zero(T), 2.5, 0.7) == 1.0
        end
    end

    @testset "primal_weight_init" begin
        lp, _ = CoolPDLP.random_milp_and_sol(5, 10, 0.4)
        params = CoolPDLP.StepSizeParameters(; invnorm_scaling = 0.9, primal_weight_damping = 0.5, zero_tol = 1.0e-10)
        @test CoolPDLP.primal_weight_init(lp, params) == 1.0

        qp = linear_quadratic_qp()
        ω_qp = CoolPDLP.primal_weight_init(qp, params)
        expected_ω = norm(qp.c) / norm(CoolPDLP.combine.(qp.lc, qp.uc))
        @test ω_qp ≈ expected_ω
    end

    @testset "update_step_size! LP is no-op" begin
        lp, _ = CoolPDLP.random_milp_and_sol(5, 10, 0.4)
        params = CoolPDLP.StepSizeParameters(; invnorm_scaling = 0.9, primal_weight_damping = 0.5, zero_tol = 1.0e-10)
        step_sizes = CoolPDLP.StepSizes(; η = 0.1, ω = 2.0, norm_A = 1.0, norm_Q = 0.0)
        η0, ω0 = step_sizes.η, step_sizes.ω
        CoolPDLP.update_step_size!(step_sizes, lp, params)
        @test step_sizes.η == η0
        @test step_sizes.ω == ω0
    end

    @testset "update_step_size! QP recomputes η" begin
        qp = simple_equality_qp()
        params = CoolPDLP.StepSizeParameters(; invnorm_scaling = 0.9, primal_weight_damping = 0.5, zero_tol = 1.0e-10)
        step_sizes = CoolPDLP.StepSizes(; η = 999.0, ω = 1.0, norm_A = 1.0, norm_Q = 1.0)
        CoolPDLP.update_step_size!(step_sizes, qp, params)
        @test step_sizes.η ≈ CoolPDLP.compute_eta(1.0, 1.0, 1.0, 0.9)
    end
end
