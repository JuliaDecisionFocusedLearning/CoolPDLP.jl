using Adapt
using CoolPDLP
using GPUArraysCore
using JLArrays
using KernelAbstractions
using LinearAlgebra
using Random: shuffle
using SparseArrays
using StableRNGs: StableRNG
using Test

A_candidates = [
    sprand(m, n, p)
        for m in (10, 20, 30)
        for n in (10, 20, 30)
        for p in (0.01, 0.1, 0.2, 0.3)
];
b_candidates = [rand(size(A, 2)) for A in A_candidates];
c_candidates = [rand(size(A, 1)) for A in A_candidates];
α, β = rand(), rand()

function test_sparse_matrix(::Type{M}; A, b, c, α, β) where {M}
    A_jl = adapt(JLBackend(), M(A))
    At_jl = adapt(JLBackend(), M(sparse(transpose(A))))
    b_jl, c_jl = jl(b), jl(c)
    @test @allowscalar Matrix(A_jl) == A
    @test @allowscalar SparseMatrixCSC(A_jl) == A
    @test nnz(A_jl) == nnz(A)
    @test get_backend(A_jl) isa JLBackend
    @test mul!(copy(c_jl), A_jl, b_jl, α, β) ≈ mul!(copy(c), A, b, α, β)
    @test @allowscalar Matrix(CoolPDLP.sametype_transpose(A_jl)) == transpose(A)
    @test typeof(CoolPDLP.sametype_transpose(A_jl)) == typeof(At_jl)
    return nothing
end

@testset for M in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
    for (A, b, c) in collect(zip(A_candidates, b_candidates, c_candidates))
        test_sparse_matrix(M; A, b, c, α, β)
        # test β is a strong zero, e.g. c should never be read since it may be uninitialized and contain NaNs
        c′ = similar(c)
        fill!(c′, NaN)
        test_sparse_matrix(M; A, b, c = c′, α, β = 0.0)
        copy!(c′, c)
        test_sparse_matrix(M; A, b, c = c′, α = 1.0, β)
        fill!(c′, NaN)
        test_sparse_matrix(M; A, b, c = c′, α = 1.0, β = 0.0)
    end
end

@testset "spmm! $M" for M in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
    A = sprand(8, 6, 0.35)
    A_jl = adapt(JLBackend(), M(A))
    rhs, lhs = rand(size(A, 2), 3), rand(size(A, 1), 3)
    α, β = rand(), rand()

    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), α, β) ≈ α * A * rhs + β * lhs
    # a zero β is a strong zero, so a destination full of NaNs is never read
    nans() = jl(fill(NaN, size(lhs)))
    @test mul!(nans(), A_jl, jl(rhs), α, 0.0) ≈ α * A * rhs
    @test mul!(nans(), A_jl, jl(rhs), 1.0, 0.0) ≈ A * rhs
    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), 1.0, β) ≈ A * rhs + β * lhs
end

@testset "mul! dimension mismatch $M" for M in (
        GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL,
    )
    A = sprand(8, 6, 0.35)
    A_jl = adapt(JLBackend(), M(A))
    b, c = jl(rand(6)), jl(rand(8))
    @test_throws DimensionMismatch mul!(c, A_jl, jl(rand(5)), 1.0, 0.0)
    @test_throws DimensionMismatch mul!(jl(rand(7)), A_jl, b, 1.0, 0.0)
    rhs, lhs = jl(rand(6, 3)), jl(rand(8, 3))
    @test_throws DimensionMismatch mul!(lhs, A_jl, jl(rand(5, 3)), 1.0, 0.0)
    @test_throws DimensionMismatch mul!(jl(rand(7, 3)), A_jl, rhs, 1.0, 0.0)
end

@testset "GPUSparseMatrixELL with zero rows" begin
    A = spzeros(0, 5)
    A_ell = GPUSparseMatrixELL(A)
    @test size(A_ell) == (0, 5)
    @test SparseMatrixCSC(A_ell) == A
end

# `spmv_coo!` sums a run of same-row nonzeros before touching `c`, so the constructor stores
# them row by row. Correctness must not depend on that ordering, only speed.
@testset "COO groups nonzeros by row" begin
    rng = StableRNG(0)
    @testset "$label" for (label, A) in (
            "short rows" => sprand(rng, 401, 260, 0.02),
            "long rows" => sprand(rng, 120, 300, 0.4),
            "one dense row among sparse ones" =>
                sparse(vcat(fill(7, 300), 1:120), vcat(1:300, fill(2, 120)), 1.0, 120, 300),
            "empty trailing rows" => sparse([1, 2], [1, 3], [2.0, 3.0], 40, 40),
            "no nonzeros at all" => spzeros(30, 20),
        )
        A_coo = GPUSparseMatrixCOO(A)
        @test issorted(A_coo.rowval)
        @test SparseMatrixCSC(A_coo) == A
        At = CoolPDLP.sametype_transpose(A_coo)
        @test issorted(At.rowval)
        @test SparseMatrixCSC(At) == SparseMatrixCSC(transpose(A))

        A_jl = adapt(JLBackend(), A_coo)
        b, c = rand(rng, size(A, 2)), rand(rng, size(A, 1))
        @test mul!(jl(copy(c)), A_jl, jl(b), α, β) ≈ α * (A * b) + β * c
        # a zero β is a strong zero, so a destination full of NaNs is never read
        @test mul!(jl(fill(NaN, size(A, 1))), A_jl, jl(b), 1.0, 0.0) ≈ A * b
        B, C = rand(rng, size(A, 2), 3), rand(rng, size(A, 1), 3)
        @test mul!(jl(copy(C)), A_jl, jl(B), α, β) ≈ α * (A * B) + β * C
        @test mul!(jl(fill(NaN, size(A, 1), 3)), A_jl, jl(B), 1.0, 0.0) ≈ A * B

        @testset "an unsorted matrix still multiplies correctly" begin
            I, J, V = findnz(A)
            perm = shuffle(rng, collect(eachindex(V)))
            shuffled = adapt(
                JLBackend(),
                GPUSparseMatrixCOO(size(A, 1), size(A, 2), I[perm], J[perm], V[perm])
            )
            @test mul!(jl(fill(NaN, size(A, 1))), shuffled, jl(b), 1.0, 0.0) ≈ A * b
        end
    end
end
