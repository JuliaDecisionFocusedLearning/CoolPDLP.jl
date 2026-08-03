using Metal
using Test

@info "Running Metal tests"

@testset "MOI" begin
    include("../moi.jl")
    test_moi(CoolPDLP.GPUSparseMatrixCSR, MetalBackend())
end
