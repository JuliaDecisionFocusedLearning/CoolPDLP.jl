"""
    PresolveStatus

Status returned by [`apply_presolve`](@ref).
"""
@enum PresolveStatus begin
    PresolveUnchanged             = 0
    PresolveReduced               = 1
    PresolveInfeasible            = 2
    PresolveUnbounded             = 3
    PresolveUnboundedOrInfeasible = 4
end

"""Abstract supertype for all presolve algorithms."""
abstract type AbstractPresolver end

"""No-op presolver."""
struct NoPresolver <: AbstractPresolver end

"""Abstract supertype for handles returned by [`apply_presolve`](@ref)."""
abstract type AbstractPresolveResult end

"""
    apply_presolve(presolver, milp::MILP) -> (status, result)

Apply `presolver` to `milp`.  Returns a 2-tuple:

- `status::PresolveStatus` — outcome of the presolve.
- `result::AbstractPresolveResult` — opaque handle used to map warm starts and
  recover solutions.  `result.milp_to_solve` is the MILP that
  will be passed to the solver (the reduced problem when
  `status == PresolveReduced`, the original otherwise).

When `status` indicates infeasibility or unboundedness `result.milp_to_solve`
is a placeholder and is not used.
"""
function apply_presolve end

"""
    map_warmstart(result, sol::PrimalDualSolution) -> PrimalDualSolution

Map `sol` from the original space into the reduced space to use as a warm start.
"""
function map_warmstart end

"""
    recover_solution(result, reduced_sol::PrimalDualSolution) -> PrimalDualSolution

Reconstruct the original-problem solution from the reduced-problem solution.
"""
function recover_solution end
