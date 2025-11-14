using Adapt
using CoolPDLP
using KernelAbstractions
using JLArrays
using SparseArrays
using Test

netlib = list_netlib_instances()

milp = read_netlib_instance(netlib[4])
sad = SaddlePointProblem(milp)
sad_32 = single_precision(sad)
sad_device = set_matrix_type(GPUSparseMatrixCSR, sad_32)
sad_jl = adapt(JLBackend(), sad_device);

@test get_backend(sad) isa CPU
@test get_backend(sad_jl) isa JLBackend

@test sad isa SaddlePointProblem{
    Float64,
    Vector{Float64},
    SparseMatrixCSC{Float64, Int64},
}
@test sad_32 isa SaddlePointProblem{
    Float32,
    Vector{Float32},
    SparseMatrixCSC{Float32, Int32},
}
@test sad_device isa SaddlePointProblem{
    Float32,
    Vector{Float32},
    GPUSparseMatrixCSR{
        Float32, Int32,
        Vector{Float32}, Vector{Int32},
    },
}
@test sad_jl isa SaddlePointProblem{
    Float32,
    JLVector{Float32},
    GPUSparseMatrixCSR{
        Float32, Int32,
        JLVector{Float32}, JLVector{Int32},
    },
}
