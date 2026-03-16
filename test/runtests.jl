using Test
using Preferences: set_preferences!
# see https://github.com/MilesCranmer/DispatchDoctor.jl?tab=readme-ov-file#-usage-in-packages
set_preferences!("CoolPDLP", "default_codegen_level" => "min")

@testset verbose = true "CoolPDLP" begin
    moisuite = get(ENV, "COOLPDLP_TEST_MOISUITE", nothing)
    if moisuite != "1"
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
    end
    if moisuite != "0"
        @testset "MOI Wrapper" begin
            include("moi.jl")
        end
    end
end
