using CoolPDLP
using JLArrays
using KernelAbstractions
using Test

@testset "Common backend" begin
    @test CoolPDLP.common_backend(rand(2), rand(4)) == CPU()
    @test CoolPDLP.common_backend(jl(rand(2)), jl(rand(4)), jl(rand(6))) == JLBackend()
    @test_throws ArgumentError CoolPDLP.common_backend(rand(2), jl(rand(4)), jl(rand(6)))
end
