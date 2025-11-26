module CoolPDLP

using Adapt
using Atomix
using DispatchDoctor
using DocStringExtensions
using GPUArrays
using IterativeSolvers
using KernelAbstractions
using LinearAlgebra
using MathOptInterface: TerminationStatusCode, ITERATION_LIMIT, OPTIMAL, TIME_LIMIT
using OhMyThreads
using ProgressMeter
using QPSReader
using QPSReader: VTYPE_Binary, VTYPE_Integer
using Random
using SparseArrays
using StableRNGs
using Statistics

@stable begin
    include("utils/matrices.jl")
    include("utils/linalg.jl")
    include("utils/device.jl")

    include("problems/milp.jl")
    include("problems/modify.jl")
    include("problems/solution.jl")

    include("components/errors.jl")
    include("components/termination.jl")
    include("components/restart.jl")
    include("components/preconditioning.jl")
    include("components/permutation.jl")
    include("components/step_size.jl")
    include("components/generic.jl")
    include("components/parameters.jl")

    include("algorithms/pdhg.jl")
end

export GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL

export MILP, nbvar, nbvar_int, nbvar_cont, nbcons, nbcons_eq, nbcons_ineq
export set_eltype, set_indtype, single_precision, set_matrix_type
export PrimalDualSolution

export PDHGParameters, PDHGState, pdhg
export is_feasible, objective_value

end # module CoolPDLP
