using CUDA

@info "Running CUDA tests"
@test CUDA.functional()
CUDA.versioninfo()

@testset "Matrices" begin
    include("matrices.jl")
end
@testset verbose = true "Batching" begin
    import cuSPARSE
    using CoolPDLP: GPUSparseMatrixCSR
    include("../gpu/batching.jl")
    @testset "$M" for M in (GPUSparseMatrixCSR, cuSPARSE.CuSparseMatrixCSR)
        test_batching(M, CUDABackend())
    end
end
@testset "MOI" begin
    import cuSPARSE
    include("../gpu/moi.jl")
    test_moi(cuSPARSE.CuSparseMatrixCSR, CUDABackend())
end
