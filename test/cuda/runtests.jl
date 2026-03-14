using CUDA

@info "Running CUDA tests"
@test CUDA.functional()
CUDA.versioninfo()

@testset "MOI" begin
    include("moi.jl")
end
