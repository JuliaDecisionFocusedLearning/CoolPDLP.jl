module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, ConversionParameters, MILP, PrimalDualSolution, PaPILOPresolver,
    milp_to_mps, mps_to_milp, perform_conversion, write_sol_file, read_sol_file
using DocStringExtensions: TYPEDFIELDS
using LinearAlgebra: dot
using PaPILO: PaPILO

"""
    PaPILOPresolveState

The `state` object produced by `presolve(::PaPILOPresolver, milp)` and consumed by
`postsolve(::PaPILOPresolver, state, sol_reduced, conversion)`. It lives in the `CoolPDLPPaPILOExt`
extension rather than in `CoolPDLP` itself, since nothing outside this backend needs it.

# Fields

$(TYPEDFIELDS)
"""
struct PaPILOPresolveState{V <: AbstractVector, S <: PrimalDualSolution}
    "path to the postsolve archive written by PaPILO"
    postsolve_file::String
    "variable names of the original problem (as they appear in the input MPS file)"
    var_names_orig::Vector{String}
    "variable names of the presolved problem (as they appear in the reduced MPS file)"
    var_names_reduced::Vector{String}
    "objective vector of the presolved problem, to report its objective value to PaPILO"
    c_reduced::V
    "zero solution of the original problem, giving `postsolve` its shape and its host containers"
    sol_orig_proto::S
end

"""
    presolve(presolver::PaPILOPresolver, milp) -> (milp_reduced, state)

Write `milp` to a temporary MPS file, run PaPILO's presolve command, and read the (typically
smaller) reduced problem back as a CPU-`Float64` `MILP` (`solve` converts it).

Raises on failure — the generic fallback-vs-error handling lives in `CoolPDLP`'s top-level
`solve`, driven by `PresolveParameters.strict`, not here.
"""
function CoolPDLP.presolve(presolver::PaPILOPresolver, milp::MILP)
    input_file = tempname() * ".mps"
    postsolve_file = tempname() * ".postsolve"
    reduced_file = tempname() * ".mps"
    try
        milp_to_mps(milp, input_file)
        if presolver.verbose
            PaPILO.presolve_write_from_file(input_file, postsolve_file, reduced_file)
        else
            redirect_stdout(devnull) do
                return PaPILO.presolve_write_from_file(input_file, postsolve_file, reduced_file)
            end
        end
        milp_reduced = mps_to_milp(
            reduced_file; dataset = milp.dataset,
            name = string(milp.name, " [presolved with PaPILO]"), path = milp.path,
        )
        state = PaPILOPresolveState(
            postsolve_file,
            milp.var_names,
            milp_reduced.var_names,
            Array(milp_reduced.c),
            PrimalDualSolution(milp),
        )
        return milp_reduced, state
    finally
        isfile(input_file) && rm(input_file; force = true)
        isfile(reduced_file) && rm(reduced_file; force = true)
    end
end

"""
    postsolve(presolver::PaPILOPresolver, state, sol_reduced, conversion) -> PrimalDualSolution

Write `sol_reduced`'s primal part to a plain-text solution file, run PaPILO's postsolve command,
and read the original-space primal solution back, typed for `conversion` by
`perform_conversion`.

The dual part is filled with `NaN` since PaPILO's file-based interface does not round-trip dual
solutions (see `CoolPDLP.postsolve`'s docstring).
"""
function CoolPDLP.postsolve(
        presolver::PaPILOPresolver, state::PaPILOPresolveState,
        sol_reduced::PrimalDualSolution, conversion::ConversionParameters
    )
    reduced_sol_file = tempname() * ".sol"
    original_sol_file = tempname() * ".sol"
    try
        x_reduced = Array(sol_reduced.x)
        write_sol_file(
            reduced_sol_file, x_reduced, state.var_names_reduced, dot(state.c_reduced, x_reduced)
        )
        if presolver.verbose
            PaPILO.postsolve_from_file(state.postsolve_file, reduced_sol_file, original_sol_file)
        else
            redirect_stdout(devnull) do
                return PaPILO.postsolve_from_file(state.postsolve_file, reduced_sol_file, original_sol_file)
            end
        end
        x_orig = read_sol_file(original_sol_file, state.var_names_orig)
        proto = state.sol_orig_proto
        x = copyto!(similar(proto.x), x_orig)
        y = fill!(similar(proto.y), NaN)
        return perform_conversion(PrimalDualSolution(x, y), conversion)
    finally
        isfile(reduced_sol_file) && rm(reduced_sol_file; force = true)
        isfile(original_sol_file) && rm(original_sol_file; force = true)
        isfile(state.postsolve_file) && rm(state.postsolve_file; force = true)
    end
end

end
