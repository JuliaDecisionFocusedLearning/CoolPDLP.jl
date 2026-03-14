using Pkg
using Test

GROUP = get(ENV, "COOLPDLP_TEST_GROUP", nothing)

@testset verbose = true "CoolPDLP" begin
    if GROUP == "Core" || isnothing(GROUP)
        @testset "Formalities" begin
            include("formalities.jl")
        end
        @testset "Tutorial" begin
            include("tutorial.jl")
        end
        for folder in readdir(@__DIR__)
            isdir(joinpath(@__DIR__, folder)) || continue
            startswith(folder, "cuda") && continue
            @testset verbose = true "$folder" begin
                for file in readdir(joinpath(@__DIR__, folder))
                    @testset "$file" begin
                        include(joinpath(@__DIR__, folder, file))
                    end
                end
            end
        end
    end
    if GROUP == "MOI" || isnothing(GROUP)
        @testset "MOI Wrapper" begin
            include("moi.jl")
        end
    end
    if GROUP == "CUDA"  # don't test this if GROUP is not specified
        Pkg.add("CUDA")
        @testset "CUDA" begin
            include("cuda/runtests.jl")
        end
    end

end
