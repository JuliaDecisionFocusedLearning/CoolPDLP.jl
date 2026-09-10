using Adapt
using CoolPDLP
using GPUArraysCore
using JLArrays
using KernelAbstractions
using LinearAlgebra
using SparseArrays
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

# `mul!` on a `GPUSparseMatrixCSR` picks a kernel from the matrix's average row length, so
# the matrices above (at most nine nonzeros per row) only reach the narrow sub-groups. These
# build one matrix per branch instead of hoping a random density lands in it.
"""
    banded_csr(m, n, nz_per_row)

Sparse `m x n` matrix with exactly `nz_per_row` nonzeros in every row, at columns spread
across the row so the kernels do a real gather rather than a contiguous read.
"""
function banded_csr(m, n, nz_per_row)
    stride = max(n ÷ nz_per_row, 1)
    I = repeat(1:m, inner = nz_per_row)
    J = [mod1(1 + ((i + k) * stride) % n, n) for i in 1:m for k in 1:nz_per_row]
    V = [float(i + k) for i in 1:m for k in 1:nz_per_row]
    return sparse(I, J, V, m, n, +)
end

@testset "CSR sub-group sizes" begin
    # row counts are deliberately not multiples of the workgroup size, so the padded launch
    # has to discard its trailing work items
    @testset "$nz_per_row nonzeros per row" for (m, n, nz_per_row, expected) in (
            (1013, 40, 1, 1),
            (1013, 40, 3, 2),
            (1013, 40, 5, 4),
            (1013, 60, 11, 8),
            (523, 80, 20, 16),
            (523, 150, 41, 32),
            (523, 150, 130, 32),
        )
        A = banded_csr(m, n, nz_per_row)
        A_jl = adapt(JLBackend(), GPUSparseMatrixCSR(A))
        @test CoolPDLP.subgroup_size(A_jl) == expected
        b, c = rand(n), rand(m)
        @test mul!(jl(copy(c)), A_jl, jl(b), α, β) ≈ α * (A * b) + β * c
        @test mul!(jl(fill(NaN, m)), A_jl, jl(b), 1.0, 0.0) ≈ A * b
        B, C = rand(n, 3), rand(m, 3)
        @test mul!(jl(copy(C)), A_jl, jl(B), α, β) ≈ α * (A * B) + β * C
        @test mul!(jl(fill(NaN, m, 3)), A_jl, jl(B), 1.0, 0.0) ≈ A * B
    end

    # a few very long rows next to empty ones: the shape that sub-groups exist to handle,
    # and the one where a row spans several passes of the strided inner loop
    @testset "skewed rows" begin
        m, n = 401, 260
        A = banded_csr(m, n, 6)
        A[3, :] .= 0
        A[4, :] .= 0
        A[7, :] = 1:n
        A[m, :] = 1:n
        A = sparse(A)
        A_jl = adapt(JLBackend(), GPUSparseMatrixCSR(A))
        b, c = rand(n), rand(m)
        @test mul!(jl(copy(c)), A_jl, jl(b), α, β) ≈ α * (A * b) + β * c
        @test mul!(jl(fill(NaN, m)), A_jl, jl(b), 1.0, 0.0) ≈ A * b
        B = rand(n, 2)
        @test mul!(jl(fill(NaN, m, 2)), A_jl, jl(B), 1.0, 0.0) ≈ A * B
    end

    @testset "no nonzeros at all" begin
        A = spzeros(37, 21)
        A_jl = adapt(JLBackend(), GPUSparseMatrixCSR(A))
        @test CoolPDLP.subgroup_size(A_jl) == 1
        @test mul!(jl(fill(NaN, 37)), A_jl, jl(rand(21)), 1.0, 0.0) ≈ zeros(37)
    end
end

@testset "CSR sub-group shrinks with the batch" begin
    # a sub-group creates parallelism, which a batch dimension already supplies, so the
    # choice has to come down as the batch grows -- and the results must not move with it
    A = banded_csr(1013, 60, 20)
    A_jl = adapt(JLBackend(), GPUSparseMatrixCSR(A))
    @test CoolPDLP.subgroup_size(A_jl, 1) == CoolPDLP.subgroup_size(A_jl)
    sizes = [CoolPDLP.subgroup_size(A_jl, nb) for nb in (1, 32, 1024, 32768)]
    @test issorted(sizes; rev = true)
    @test sizes[end] < sizes[1]
    @test all(>=(1), sizes)

    # a wide batch may only narrow the sub-group, never widen it: the bound on how far it
    # narrows is not a floor on the result
    @testset "$nz_per_row nonzeros per row" for nz_per_row in (1, 2, 3, 5, 11, 20, 60, 130)
        A_short = adapt(JLBackend(), GPUSparseMatrixCSR(banded_csr(997, 200, nz_per_row)))
        plain = CoolPDLP.subgroup_size(A_short)
        @test all(
            CoolPDLP.subgroup_size(A_short, nb) <= plain for nb in (1, 10, 100, 10^4, 10^6)
        )
    end
    @testset "nbatch=$nbatch" for nbatch in (1, 2, 32, 40, 1024)
        B = rand(size(A, 2), nbatch)
        C = rand(size(A, 1), nbatch)
        @test mul!(jl(copy(C)), A_jl, jl(B), α, β) ≈ α * (A * B) + β * C
        @test mul!(jl(fill(NaN, size(A, 1), nbatch)), A_jl, jl(B), 1.0, 0.0) ≈ A * B
    end
end
