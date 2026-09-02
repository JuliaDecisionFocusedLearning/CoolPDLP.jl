module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PaPILOPresolver,
    milp_to_mps, mps_to_milp, write_sol_file, read_sol_file
using DocStringExtensions: TYPEDFIELDS
using LinearAlgebra: dot
using PaPILO: PaPILO

"""
    PaPILOPresolveState

The `state` object produced by `presolve(::PaPILOPresolver, milp)` and consumed by
`postsolve(::PaPILOPresolver, state, sol_reduced)`. It lives in the `CoolPDLPPaPILOExt`
extension rather than in `CoolPDLP` itself, since nothing outside this backend needs it.

# Fields

$(TYPEDFIELDS)
"""
struct PaPILOPresolveState{S <: PrimalDualSolution}
    "path to the postsolve archive written by PaPILO"
    postsolve_file::String
    "variable names of the original problem (as they appear in the input MPS file)"
    var_names_orig::Vector{String}
    "variable names of the presolved problem (as they appear in the reduced MPS file)"
    var_names_reduced::Vector{String}
    "objective vector of the presolved problem, to report its objective value to PaPILO"
    c_reduced::Vector{Float64}
    "zero solution of the original problem, whose element and array types `postsolve` reproduces"
    sol_orig_proto::S
end

"""
    presolve(presolver::PaPILOPresolver, milp) -> (milp_reduced, state)

Write `milp` to a temporary MPS file, run PaPILO's presolve command, and read the (typically
smaller) reduced problem back, with the same element and array types as `milp`.

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
            reduced_file, milp; dataset = milp.dataset,
            name = string(milp.name, "[presolved with PaPILO]"), path = milp.path,
        )
        state = PaPILOPresolveState(
            postsolve_file,
            milp.var_names,
            milp_reduced.var_names,
            Vector{Float64}(Array(milp_reduced.c)),
            PrimalDualSolution(milp),
        )
        return milp_reduced, state
    finally
        isfile(input_file) && rm(input_file; force = true)
        isfile(reduced_file) && rm(reduced_file; force = true)
    end
end

"""
    postsolve(presolver::PaPILOPresolver, state, sol_reduced) -> PrimalDualSolution

Write `sol_reduced`'s primal part to a plain-text solution file, run PaPILO's postsolve command,
and read the original-space primal solution back, with the same element and array types as a
solution of the original problem.

The dual part is filled with `NaN` since PaPILO's file-based interface does not round-trip dual
solutions (see `CoolPDLP.postsolve`'s docstring).
"""
function CoolPDLP.postsolve(
        presolver::PaPILOPresolver, state::PaPILOPresolveState, sol_reduced::PrimalDualSolution
    )
    reduced_sol_file = tempname() * ".sol"
    original_sol_file = tempname() * ".sol"
    try
        x_reduced = Vector{Float64}(Array(sol_reduced.x))
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
        T = eltype(proto.x)
        x = copyto!(similar(proto.x), map(T, x_orig))
        y = fill!(similar(proto.y), T(NaN))
        return PrimalDualSolution(x, y)
    finally
        isfile(reduced_sol_file) && rm(reduced_sol_file; force = true)
        isfile(original_sol_file) && rm(original_sol_file; force = true)
        isfile(state.postsolve_file) && rm(state.postsolve_file; force = true)
    end
end

end
