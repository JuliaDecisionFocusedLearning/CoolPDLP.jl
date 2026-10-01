# Reactant tracing needs CUDA.jl to compile the kernels, on any backend
using CoolPDLP
using CoolPDLP: GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL, termination_status
using LinearAlgebra
using Random
using Reactant
using Reactant: to_rarray
using SparseArrays
using Test

Reactant.set_default_backend("gpu")
@test lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())) != "cpu"

const MATRIX_TYPES = (GPUSparseMatrixCSR, GPUSparseMatrixELL, GPUSparseMatrixCOO)

@testset verbose = true "Compiled products" begin
    rng = Xoshiro(0)
    A_cpu = sprandn(rng, 24, 16, 0.3)
    α, β = 0.7, 0.3

    @testset "$M" for M in MATRIX_TYPES
        A_r = to_rarray(M(A_cpu); track_numbers = true)

        @testset "vector" begin
            b, c0 = randn(rng, 16), randn(rng, 24)
            b_r, c_r = to_rarray(b), to_rarray(copy(c0))
            compiled = @compile mul!(c_r, A_r, b_r, α, β)
            compiled(c_r, A_r, b_r, α, β)
            @test Array(c_r) ≈ α * (A_cpu * b) + β * c0
        end

        @testset "vector, row-skewed" begin
            # a row far longer than the others sends a CSR product to `spmv_csr_blocks!`
            m, n = 1024, 400
            A_skew = sparse([1:m; fill(5, n)], [mod1.(1:m, n); 1:n], randn(rng, m + n), m, n)
            M == GPUSparseMatrixCSR && @test !isempty(M(A_skew).blocks)
            A_skew_r = to_rarray(M(A_skew); track_numbers = true)
            b, c0 = randn(rng, n), randn(rng, m)
            b_r, c_r = to_rarray(b), to_rarray(copy(c0))
            compiled = @compile mul!(c_r, A_skew_r, b_r, α, β)
            compiled(c_r, A_skew_r, b_r, α, β)
            @test Array(c_r) ≈ α * (A_skew * b) + β * c0
        end

        @testset "batch of vectors" begin
            b, c0 = randn(rng, 16, 3), randn(rng, 24, 3)
            b_r, c_r = to_rarray(b), to_rarray(copy(c0))
            # refused until EnzymeAD/Reactant.jl#3269 is solved
            @test_throws "Reactant.jl/issues/3269" @compile mul!(c_r, A_r, b_r, α, β)
        end
    end
end

@testset verbose = true "Compiled solve" begin
    milp0, sol0 = CoolPDLP.random_milp_and_sol(Xoshiro(0), 20, 30, 0.4)

    # the batch rescales the objective of the same problem, one factor per column
    milp_batch = MILP(;
        c = stack(s * milp0.c for s in (1.0, 1.01, 0.99)),
        lv = repeat(milp0.lv, 1, 3), uv = repeat(milp0.uv, 1, 3), milp0.A,
        lc = repeat(milp0.lc, 1, 3), uc = repeat(milp0.uc, 1, 3), milp0.int_var,
    )
    sol_batch = PrimalDualSolution(milp_batch)

    function traced_problem(milp_init, sol_init, algo)
        milp, sol = preprocess(milp_init, sol_init, algo)
        state = initialize(milp, sol, algo; starting_time = time())
        return (
            to_rarray(state; track_numbers = true),
            to_rarray(milp; track_numbers = true),
            to_rarray(algo; track_numbers = true),
        )
    end

    @testset "$M" for M in MATRIX_TYPES
        algo = PDLP(
            Float64,
            Int32,
            M;
            backend = nothing,
            termination_reltol = 1.0e-6,
            max_kkt_passes = 200,
            time_limit = Inf,
            check_every = 50,
            record_error_history = false,
            show_progress = false,
        )

        @testset "single" begin
            milp, sol = preprocess(milp0, sol0, algo)
            state = initialize(milp, sol, algo; starting_time = time())
            CoolPDLP.solve!(state, milp, algo)

            state_r, milp_r, algo_r = traced_problem(milp0, sol0, algo)
            compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)
            compiled_solve!(state_r, milp_r, algo_r)

            @test all(isfinite, Array(state_r.sol.x))
            @test Int(state_r.stats.kkt_passes) == state.stats.kkt_passes
            @test termination_status(state_r.stats) == termination_status(state.stats)
            @test Array(state_r.sol.x) ≈ Array(state.sol.x) rtol = 1.0e-6
            @test Array(state_r.sol.y) ≈ Array(state.sol.y) rtol = 1.0e-6
        end

        @testset "batched" begin
            # refused until EnzymeAD/Reactant.jl#3269 is solved
            state_r, milp_r, algo_r = traced_problem(milp_batch, sol_batch, algo)
            @test_throws "Reactant.jl/issues/3269" @compile CoolPDLP.solve!(
                state_r, milp_r, algo_r
            )
        end
    end
end
