using Metal
using Test

@info "Running Metal tests"

@testset verbose = true "Batching" begin
    include("../batching.jl")
    test_batching(CoolPDLP.GPUSparseMatrixCSR, MetalBackend(), Float32)
end
@testset "MOI" begin
    include("../moi.jl")
    test_moi(CoolPDLP.GPUSparseMatrixCSR, MetalBackend(), Float32)
end
