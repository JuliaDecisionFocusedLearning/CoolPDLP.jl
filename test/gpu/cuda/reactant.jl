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

        @testset "batch of vectors" begin
            b, c0 = randn(rng, 16, 3), randn(rng, 24, 3)
            b_r, c_r = to_rarray(b), to_rarray(copy(c0))
            compiled = @compile mul!(c_r, A_r, b_r, α, β)
            compiled(c_r, A_r, b_r, α, β)
            @test Array(c_r) ≈ α * (A_cpu * b) + β * c0
        end
    end
end

# batched solves hit the XLA layout bug that `flat_mul!` works around
@testset verbose = true "Compiled solve" begin
    milp0, sol0 = CoolPDLP.random_milp_and_sol(Xoshiro(0), 20, 30, 0.4)

    # the batch rescales the objective of the same problem, one factor per column
    milp_batch = MILP(;
        c = stack(s * milp0.c for s in (1.0, 1.01, 0.99)),
        lv = repeat(milp0.lv, 1, 3), uv = repeat(milp0.uv, 1, 3), milp0.A,
        lc = repeat(milp0.lc, 1, 3), uc = repeat(milp0.uc, 1, 3), milp0.int_var,
    )
    sol_batch = PrimalDualSolution(milp_batch)

    @testset "$M, $(batched ? "batched" : "single")" for M in MATRIX_TYPES, batched in (false, true)
        milp_init, sol_init = batched ? (milp_batch, sol_batch) : (milp0, sol0)
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

        milp, sol = preprocess(milp_init, sol_init, algo)
        state = initialize(milp, sol, algo; starting_time = time())
        CoolPDLP.solve!(state, milp, algo)

        milp_copy, sol_copy = preprocess(milp_init, sol_init, algo)
        state_copy = initialize(milp_copy, sol_copy, algo; starting_time = time())
        milp_r = to_rarray(milp_copy; track_numbers = true)
        state_r = to_rarray(state_copy; track_numbers = true)
        algo_r = to_rarray(algo; track_numbers = true)
        compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)
        compiled_solve!(state_r, milp_r, algo_r)

        @test all(isfinite, Array(state_r.sol.x))
        @test Int(state_r.stats.kkt_passes) == state.stats.kkt_passes
        @test termination_status(state_r.stats) == termination_status(state.stats)
        @test Array(state_r.sol.x) ≈ Array(state.sol.x) rtol = 1.0e-6
        @test Array(state_r.sol.y) ≈ Array(state.sol.y) rtol = 1.0e-6
    end
end
