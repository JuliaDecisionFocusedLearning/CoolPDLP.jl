using CUDA

@info "Running CUDA tests"
@test CUDA.functional()
CUDA.versioninfo()

@testset "Matrices" begin
    include("matrices.jl")
end
@testset "MOI" begin
    import cuSPARSE
    include("../gpu/moi.jl")
    test_moi(cuSPARSE.CuSparseMatrixCSR, CUDABackend())
end
