using Pkg
using Test
using Preferences: set_preferences!
# see https://github.com/MilesCranmer/DispatchDoctor.jl?tab=readme-ov-file#-usage-in-packages
set_preferences!("CoolPDLP", "default_codegen_level" => "min")

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
    # don't test this if GROUP is not specified
    if GROUP == "Perf"
        # test separately in CI to avoid Codecov noise
        @testset "Performance" begin
            include("perf.jl")
        end
    end
    if GROUP == "CUDA"
        Pkg.add(
            [
                Pkg.PackageSpec(name = "CUDA", rev = "main"),
                Pkg.PackageSpec(name = "CUDACore", rev = "main"),
                Pkg.PackageSpec(name = "CUDATools", rev = "main"),
                Pkg.PackageSpec(name = "CUPTI", rev = "main"),
                Pkg.PackageSpec(name = "NVML", rev = "main"),
                Pkg.PackageSpec(name = "cuBLAS", rev = "main"),
                Pkg.PackageSpec(name = "cuSPARSE", rev = "main"),
                Pkg.PackageSpec(name = "cuSOLVER", rev = "main"),
                Pkg.PackageSpec(name = "cuFFT", rev = "main"),
                Pkg.PackageSpec(name = "cuRAND", rev = "main"),
                Pkg.PackageSpec(name = "cuDNN", rev = "main"),
                Pkg.PackageSpec(name = "cuTENSOR", rev = "main"),
                Pkg.PackageSpec(name = "cuTensorNet", rev = "main"),
                Pkg.PackageSpec(name = "cuStateVec", rev = "main"),
                Pkg.PackageSpec(name = "GPUCompiler", rev = "main"),
            ]
        )
        @testset verbose = true "CUDA" begin
            include("cuda/runtests.jl")
        end
    end

end
