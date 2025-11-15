module CoolPDLP

using Accessors: @set
using Adapt
using Atomix
using DataDeps
using DispatchDoctor
using DocStringExtensions
using GPUArrays
using GZip
using IterativeSolvers
using KernelAbstractions
using LinearAlgebra
using Logging
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

    include("preprocessing/permutation.jl")
    include("preprocessing/preconditioner.jl")

    include("problems/abstract.jl")
    include("problems/milp.jl")
    include("problems/sad.jl")
    include("problems/solution.jl")

    include("algorithms/types.jl")
    include("algorithms/evaluate.jl")
    include("algorithms/pdhg.jl")
    include("algorithms/pdlp.jl")

    include("input/data.jl")
    include("input/io.jl")
    include("input/adapt.jl")

end

export GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL

export nbvar, nbvar_int, nbvar_cont, nbcons, nbcons_eq, nbcons_ineq
export MILP, relax
export SaddlePointProblem
export PrimalDualSolution

export TerminationReason
export PDHGParameters, PDHGState, pdhg
export PDLPParameters, PDLPState, pdlp
export is_feasible, objective_value

export read_milp, write_sol, read_sol
export list_pdlp_miplib2017_subset, read_miplib2017_instance
export list_netlib_instances, read_netlib_instance
export set_eltype, set_indtype, single_precision, set_matrix_type

end # module CoolPDLP
