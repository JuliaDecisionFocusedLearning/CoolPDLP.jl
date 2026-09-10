# Reactant compilation of CoolPDLP's own sparse formats.
#
# This lives in the `cuda` group rather than in `test/gpu/reactant/`, because Reactant lowers a
# `KernelAbstractions` kernel through CUDA.jl's GPU compiler *whatever the target backend*, so
# `using CUDA` is required here even though the same code also compiles on Reactant's CPU
# backend. The `Reactant` group deliberately does not depend on CUDA.jl, and its
# "Custom sparse wrappers can be traced" testset covers the part of this that needs no kernel.
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

# `mul!` is what Reactant gets wrong on its own: it overlays the function for every
# `AbstractMatrix` and lowers the product to a dense `stablehlo.dot_general`, which cannot work on
# a sparse wrapper. These products must come out of the compiled program just as the kernel
# computes them outside it.
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

# A whole solve, to check that nothing else in the loop objects to a sparse matrix.
#
# Only the single-instance case is checked here. A *batched* solve on this backend comes out wrong
# from its second iteration, and the cause is not in these formats:
#
#   - the same batched solve on Reactant's CPU backend agrees with the plain loop;
#   - the same batched solve with a dense `Matrix` agrees to 2.6e-16 in this very harness;
#   - a batched product is exact on this backend on its own (the testset above), and stays exact
#     when its intermediate is internal to the compiled program and when it runs inside a
#     `@trace while` loop.
#
# What is left is how Reactant compiles a `KernelAbstractions` launch interleaved with the
# broadcasts that share `step!`'s scratch space. That belongs upstream rather than here.
@testset verbose = true "Compiled solve" begin
    milp0, sol0 = CoolPDLP.random_milp_and_sol(Xoshiro(0), 20, 30, 0.4)

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

        milp, sol = preprocess(milp0, sol0, algo)
        state = initialize(milp, sol, algo; starting_time = time())
        CoolPDLP.solve!(state, milp, algo)

        # `solve!` mutates the scratch space it shares with the problem, so the compiled run
        # starts from its own copy
        milp_copy, sol_copy = preprocess(milp0, sol0, algo)
        state_copy = initialize(milp_copy, sol_copy, algo; starting_time = time())
        milp_r = to_rarray(milp_copy; track_numbers = true)
        state_r = to_rarray(state_copy; track_numbers = true)
        algo_r = to_rarray(algo; track_numbers = true)
        compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)
        compiled_solve!(state_r, milp_r, algo_r)

        @test all(isfinite, Array(state_r.sol.x))
        # both loops must stop at the same point for the iterates to be comparable at all
        @test Int(state_r.stats.kkt_passes) == state.stats.kkt_passes
        @test termination_status(state_r.stats) == termination_status(state.stats)
        @test Array(state_r.sol.x) ≈ Array(state.sol.x) rtol = 1.0e-6
        @test Array(state_r.sol.y) ≈ Array(state.sol.y) rtol = 1.0e-6
    end
end
