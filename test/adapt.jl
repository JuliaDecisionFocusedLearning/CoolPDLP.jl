using Adapt
using CoolPDLP
using KernelAbstractions
using JLArrays
using SparseArrays
using Test

netlib = list_netlib_instances()

milp, _ = read_netlib_instance(first(list_netlib_instances()))
milp_device = change_matrix_type(DeviceSparseMatrixCSR, milp);
milp_jl = adapt(JLBackend(), milp_device);

sad = SaddlePointProblem(milp)
sad_device = change_matrix_type(DeviceSparseMatrixCSR, sad)
sad_jl = adapt(JLBackend(), sad_device)

@testset "CPU" begin
    @test get_backend(milp) isa CPU
    @test change_floating_type(Float32, milp) isa MILP{
        Float32, Vector{Float32},
        SparseMatrixCSC{Float32, Int64},
    }
    @test change_integer_type(Int32, milp) isa MILP{
        Float64, Vector{Float64},
        SparseMatrixCSC{Float64, Int32},
    }
end

@testset "Device" begin
    @test get_backend(milp_device) isa CPU
    @test milp_device isa MILP{
        Float64, Vector{Float64},
        DeviceSparseMatrixCSR{
            Float64, Int64,
            Vector{Float64}, Vector{Int64},
        },
    }
    @test change_floating_type(Float32, milp_device) isa MILP{
        Float32, Vector{Float32},
        DeviceSparseMatrixCSR{
            Float32, Int64,
            Vector{Float32}, Vector{Int64},
        },
    }
    @test change_integer_type(Int32, milp_device) isa MILP{
        Float64, Vector{Float64},
        DeviceSparseMatrixCSR{
            Float64, Int32,
            Vector{Float64}, Vector{Int32},
        },
    }
end

@testset "JL" begin
    @test get_backend(milp_jl) isa JLBackend
    @test milp_jl isa MILP{
        Float64, JLVector{Float64},
        DeviceSparseMatrixCSR{
            Float64, Int64,
            JLVector{Float64}, JLVector{Int64},
        },
    }
end

@testset "Saddle point problem" begin
    @test sad isa SaddlePointProblem{
        Float64, Int64,
        Vector{Float64},
        SparseMatrixCSC{Float64, Int64},
    }
    @test sad_device isa SaddlePointProblem{
        Float64, Int64, Vector{Float64},
        DeviceSparseMatrixCSR{
            Float64, Int64,
            Vector{Float64}, Vector{Int64},
        },
    }
    @test sad_jl isa SaddlePointProblem{
        Float64, Int64, JLVector{Float64},
        DeviceSparseMatrixCSR{
            Float64, Int64,
            JLVector{Float64}, JLVector{Int64},
        },
    }
end
