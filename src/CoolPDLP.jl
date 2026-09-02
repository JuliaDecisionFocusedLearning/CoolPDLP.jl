module CoolPDLP

# external dependencies
using Adapt: Adapt, adapt
using Atomix: Atomix
using BangBang: add!!, broadcast!!
using DispatchDoctor: @stable, @unstable
using DocStringExtensions: TYPEDFIELDS
using IterativeSolvers: powm!
using JuMP: JuMP
using KernelAbstractions: KernelAbstractions, Backend, CPU, @kernel, @index, allocate, get_backend
import MathOptInterface as MOI
using ProgressMeter: ProgressUnknown, finish!, next!
using QPSReader: QPSData, VTYPE_Binary, VTYPE_Integer
using StableRNGs: StableRNG

# standard libraries
using LinearAlgebra: LinearAlgebra, Diagonal, axpby!, diag, dot, mul!, norm
using Printf: @sprintf
using Random: Random, randn!
using SparseArrays: SparseArrays, SparseMatrixCSC, AbstractSparseMatrix, findnz, nnz, nonzeros, nzrange, sparse, sprandn

include("public.jl")

@stable begin
    include("utils/device.jl")
    include("utils/mat_coo.jl")
    include("utils/mat_csr.jl")
    include("utils/mat_ell.jl")
    include("utils/linalg.jl")
    include("utils/test.jl")
    include("utils/batching.jl")

    include("problems/milp.jl")
    include("problems/solution.jl")
    include("problems/modify.jl")

    include("components/scratch.jl")
    include("components/conversion.jl")
    @unstable include("components/presolve.jl")
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

@public sametype_transpose
@public PresolveParameters, milp_to_mps, mps_to_milp, write_sol_file, read_sol_file

export AbstractPresolver, presolve, postsolve, PaPILOPresolver

export GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL

export MILP, nbvar, nbvar_int, nbvar_cont, nbcons, nbcons_eq, nbcons_ineq
export instance, nbinstances, isbatched
export PrimalDualSolution

export preprocess, initialize, solve, solve!
export PDHG, PDLP
@public Algorithm
@public KKTErrors, relative
export is_feasible, objective_value

@public Optimizer

function __init__()
    # `presolve`/`postsolve` for a `PaPILOPresolver` live in the `CoolPDLPPaPILOExt` extension,
    # so forgetting `using PaPILO` surfaces as a plain `MethodError`: point the user at the fix
    Base.Experimental.register_error_hint(MethodError) do io, exc, _argtypes, _kwargs
        if (exc.f === presolve || exc.f === postsolve) &&
                any(a -> a isa PaPILOPresolver, exc.args)
            print(
                io,
                "\nPaPILOPresolver needs PaPILO.jl to be loaded first (it is a weak dependency " *
                    "of CoolPDLP, kept optional because of its Apache-2.0 license): run " *
                    "`using PaPILO` and try again."
            )
        end
    end
    return nothing
end

end # module CoolPDLP
