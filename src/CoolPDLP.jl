module CoolPDLP

# external dependencies
using Adapt: Adapt, adapt
using Atomix: Atomix
using DispatchDoctor: @stable
using DocStringExtensions: TYPEDFIELDS
using IterativeSolvers: powm!
using KernelAbstractions: KernelAbstractions, Backend, CPU, @kernel, @index, allocate, get_backend
import MathOptInterface as MOI
using ProgressMeter: ProgressUnknown, finish!, next!
using QPSReader: QPSData, VTYPE_Binary, VTYPE_Integer
using StableRNGs: StableRNG

# standard libraries
using LinearAlgebra: LinearAlgebra, Diagonal, axpby!, diag, dot, mul!, norm
using Printf: @sprintf
using Random: randn!
using SparseArrays: SparseArrays, SparseMatrixCSC, AbstractSparseMatrix, findnz, nnz, nonzeros, nzrange, sparse, sprandn

include("public.jl")

@stable begin
    include("utils/device.jl")
    include("utils/mat_coo.jl")
    include("utils/mat_csr.jl")
    include("utils/mat_ell.jl")
    include("utils/linalg.jl")
    include("utils/test.jl")

    include("problems/milp.jl")
    include("problems/solution.jl")
    include("problems/modify.jl")

    include("components/scratch.jl")
    include("components/conversion.jl")
    include("components/preconditioning.jl")
    include("components/permutation.jl")
    include("components/step_size.jl")
    include("components/errors.jl")
    include("components/iteration.jl")
    include("components/restart.jl")
    include("components/generic.jl")
    include("components/termination.jl")

    include("algorithms/common.jl")
    include("algorithms/pdhg.jl")
    include("algorithms/pdlp.jl")
end

include("MOI_wrapper.jl")

export GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL

export MILP, nbvar, nbvar_int, nbvar_cont, nbcons, nbcons_eq, nbcons_ineq
export PrimalDualSolution

export preprocess, initialize, solve, solve!
export PDHG, PDLP
@public Algorithm
export is_feasible, objective_value

@public Optimizer

end # module CoolPDLP
