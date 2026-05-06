using CUDA

@info "Running CUDA tests"
@test CUDA.functional()
CUDA.versioninfo()

@testset "Matrices" begin
    include("matrices.jl")
end
@testset "MOI" begin
    include("moi.jl")
end
