using pocl_jll, OpenCL
using Test

@info "Running OpenCL tests"

@testset "MOI" begin
    include("../moi.jl")
    test_moi(CoolPDLP.GPUSparseMatrixCSR, OpenCLBackend())
end
