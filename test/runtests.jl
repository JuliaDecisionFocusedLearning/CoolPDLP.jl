using Pkg
using Test
using Preferences: set_preferences!
# see https://github.com/MilesCranmer/DispatchDoctor.jl?tab=readme-ov-file#-usage-in-packages
set_preferences!("CoolPDLP", "default_codegen_level" => "min")

GROUP = get(ENV, "COOLPDLP_TEST_GROUP", nothing)

# a GPU backend and the preferences it needs must be in place before anything loads CoolPDLP
if GROUP == "cuda"
    Pkg.add(["CUDA", "cuSPARSE"])
elseif GROUP == "metal"
    Pkg.add("Metal")
elseif GROUP == "OpenCL"
    set_preferences!("CoolPDLP", "dispatch_doctor_mode" => "disable")
end

# every test file draws on the same fixtures: defining them once here rather than in each
# file keeps the log free of overwritten-method warnings
include("fixtures.jl")

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
            startswith(folder, "gpu") && continue
            @testset verbose = true "$folder" begin
                for file in readdir(joinpath(@__DIR__, folder))
                    endswith(file, ".jl") || continue
                    @testset "$file" begin
                        include(joinpath(@__DIR__, folder, file))
                    end
                end
            end
        end
        @testset verbose = true "Batching on JLArrays" begin
            using CoolPDLP: GPUSparseMatrixCSR
            using JLArrays: JLBackend
            include("gpu/batching.jl")
            test_batching(GPUSparseMatrixCSR, JLBackend())
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

    # GPU backends

    if GROUP == "cuda"
        @testset verbose = true "CUDA" begin
            include("gpu/cuda/runtests.jl")
        end
    end
    if GROUP == "metal"
        @testset verbose = true "Metal" begin
            include("gpu/metal/runtests.jl")
        end
    end
    if GROUP == "OpenCL"
        @testset verbose = true "OpenCL" begin
            include("gpu/opencl/runtests.jl")
        end
    end
end
