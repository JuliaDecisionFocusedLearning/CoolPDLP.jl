using CoolPDLP
using Test

@testset verbose = true "CoolPDLP" begin
    @testset verbose = true "Formalities" begin
        include("formalities.jl")
    end
end
