using Adapt
using CoolPDLP
using CUDA
using cuSPARSE
using GPUArraysCore
using SparseArrays
using Test

A_candidates = [
    sprand(m, n, p)
        for m in (10, 20, 30)
        for n in (10, 20, 30)
        for p in (0.01, 0.1, 0.2, 0.3)
];

@testset for M in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
    for A in A_candidates
        A_gpu = M(A)
        @test @allowscalar Matrix(CoolPDLP.sametype_transpose(A_gpu)) == transpose(A)
        @test typeof(CoolPDLP.sametype_transpose(A_gpu)) == typeof(A_gpu)
    end
end
