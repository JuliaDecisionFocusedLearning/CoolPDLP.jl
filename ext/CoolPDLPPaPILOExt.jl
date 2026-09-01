module CoolPDLPPaPILOExt

using CoolPDLP:
    CoolPDLP, MILP, PrimalDualSolution, PresolveParameters, PresolveResult,
    milp_to_mps, mps_to_milp, write_papilo_solution, read_papilo_solution
using PaPILO: PaPILO

"""
    presolve_milp_impl(milp, params)

Actual implementation of `CoolPDLP.presolve_milp`, only reachable once this extension is
loaded (i.e. once `PaPILO` has been `using`d).
"""
function presolve_milp_impl(milp::MILP, params::PresolveParameters)
    input_file = postsolve_file = reduced_file = ""
    try
        input_file = tempname() * ".mps"
        postsolve_file = tempname() * ".postsolve"
        reduced_file = tempname() * ".mps"
        milp_to_mps(milp, input_file)
        if params.verbose
            PaPILO.presolve_write_from_file(input_file, postsolve_file, reduced_file)
        else
            redirect_stdout(devnull) do
                return PaPILO.presolve_write_from_file(input_file, postsolve_file, reduced_file)
            end
        end
        milp_reduced = mps_to_milp(
            reduced_file; dataset = milp.dataset, name = milp.name, path = milp.path,
        )
        result = PresolveResult(postsolve_file, milp.var_names, milp_reduced.var_names)
        return milp_reduced, result
    catch e
        @warn "Presolve failed, falling back to the original problem" exception = e
        isfile(postsolve_file) && rm(postsolve_file; force = true)
        return milp, nothing
    finally
        isfile(input_file) && rm(input_file; force = true)
        isfile(reduced_file) && rm(reduced_file; force = true)
    end
end

"""
    postsolve_solution_impl(result, sol_reduced, params)

Actual implementation of `CoolPDLP.postsolve_solution`, only reachable once this extension is
loaded (i.e. once `PaPILO` has been `using`d).
"""
function postsolve_solution_impl(
        result::PresolveResult, sol_reduced::PrimalDualSolution, params::PresolveParameters
    )
    reduced_sol_file = tempname() * ".sol"
    original_sol_file = tempname() * ".sol"
    try
        write_papilo_solution(reduced_sol_file, Array(sol_reduced.x), result.var_names_reduced)
        if params.verbose
            PaPILO.postsolve_from_file(result.postsolve_file, reduced_sol_file, original_sol_file)
        else
            redirect_stdout(devnull) do
                return PaPILO.postsolve_from_file(result.postsolve_file, reduced_sol_file, original_sol_file)
            end
        end
        x_orig = read_papilo_solution(original_sol_file, result.var_names_orig)
        return x_orig
    finally
        rm(reduced_sol_file; force = true)
        isfile(original_sol_file) && rm(original_sol_file; force = true)
        isfile(result.postsolve_file) && rm(result.postsolve_file; force = true)
    end
end

end
