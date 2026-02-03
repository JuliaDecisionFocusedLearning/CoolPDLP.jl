using Test

@testset verbose = true "CoolPDLP" begin
    @testset "Formalities" begin
        include("formalities.jl")
    end
    @testset "Tutorial" begin
        include("tutorial.jl")
    end
    for folder in readdir(@__DIR__)
        isdir(joinpath(@__DIR__, folder)) || continue
        @testset verbose = true "$folder" begin
            for file in readdir(joinpath(@__DIR__, folder))
                @testset "$file" begin
                    include(joinpath(@__DIR__, folder, file))
                end
            end
        end
    end
    @testset "MOI Wrapper" begin
        include("MOI_wrapper.jl")
    end
end
