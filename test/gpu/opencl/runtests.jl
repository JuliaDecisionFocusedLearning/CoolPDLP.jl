using pocl_jll, OpenCL
using Test

@info "Running OpenCL tests"

@testset verbose = true "Batching" begin
    include("../batching.jl")
    test_batching(CoolPDLP.GPUSparseMatrixCSR, OpenCLBackend())
end
@testset "MOI" begin
    include("../moi.jl")
    test_moi(CoolPDLP.GPUSparseMatrixCSR, OpenCLBackend())
end
