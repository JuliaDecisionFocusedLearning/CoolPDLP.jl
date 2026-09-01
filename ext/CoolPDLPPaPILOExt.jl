module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PaPILOPresolver, PaPILOPresolveState,
    milp_to_mps, mps_to_milp, write_sol_file, read_sol_file, nbcons
using PaPILO: PaPILO

"""
    papilo_presolve(presolver, milp) -> (milp_reduced, state::PaPILOPresolveState)

Implementation of `CoolPDLP.presolve` for [`PaPILOPresolver`](@ref): write `milp` to a
temporary MPS file, run PaPILO's presolve command, and read the (typically smaller) reduced
problem back. Raises on failure — the generic fallback-vs-error handling lives in `CoolPDLP`'s
top-level `solve`, driven by `PresolveParameters.strict`, not here.
"""
function papilo_presolve(presolver::PaPILOPresolver, milp::MILP)
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
            reduced_file; dataset = milp.dataset, name = milp.name, path = milp.path,
        )
        state = PaPILOPresolveState(postsolve_file, milp.var_names, milp_reduced.var_names, nbcons(milp))
        return milp_reduced, state
    finally
        isfile(input_file) && rm(input_file; force = true)
        isfile(reduced_file) && rm(reduced_file; force = true)
    end
end

"""
    papilo_postsolve(presolver, state, sol_reduced) -> PrimalDualSolution

Implementation of `CoolPDLP.postsolve` for [`PaPILOPresolver`](@ref): write `sol_reduced`'s
primal part to a plain-text solution file, run PaPILO's postsolve command, and read the
original-space primal solution back. The dual part is filled with `NaN` since PaPILO's
file-based interface does not round-trip dual solutions (see `CoolPDLP.postsolve`'s docstring).
"""
function papilo_postsolve(
        presolver::PaPILOPresolver, state::PaPILOPresolveState, sol_reduced::PrimalDualSolution
    )
    reduced_sol_file = tempname() * ".sol"
    original_sol_file = tempname() * ".sol"
    try
        write_sol_file(reduced_sol_file, Array(sol_reduced.x), state.var_names_reduced)
        if presolver.verbose
            PaPILO.postsolve_from_file(state.postsolve_file, reduced_sol_file, original_sol_file)
        else
            redirect_stdout(devnull) do
                return PaPILO.postsolve_from_file(state.postsolve_file, reduced_sol_file, original_sol_file)
            end
        end
        x_orig = read_sol_file(original_sol_file, state.var_names_orig)
        y_orig = fill(NaN, state.nbcons_orig)
        return PrimalDualSolution(x_orig, y_orig)
    finally
        rm(reduced_sol_file; force = true)
        isfile(original_sol_file) && rm(original_sol_file; force = true)
        isfile(state.postsolve_file) && rm(state.postsolve_file; force = true)
    end
end

end
