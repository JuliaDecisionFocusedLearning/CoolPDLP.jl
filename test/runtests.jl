using CoolPDLP
using Test

@testset verbose = true "CoolPDLP" begin
    @testset "Formalities" begin
        include("formalities.jl")
    end
    @testset "Matrices" begin
        include("matrices.jl")
    end
    @testset "Linear algebra" begin
        include("linalg.jl")
    end
    @testset "IO" begin
        include("io.jl")
    end
    @testset "Adapt" begin
        include("adapt.jl")
    end
    @testset "Tutorial" begin
        include("tutorial.jl")
    end
end
