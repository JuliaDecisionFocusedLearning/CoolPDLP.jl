"""
    AbstractPresolver

Supertype for presolvers, which must implement [`presolve`](@ref) and [`postsolve`](@ref).
"""
abstract type AbstractPresolver end

"""
    presolve(presolver, milp::MILP) -> (milp_reduced, presolve_info)

Reduce `milp` into a smaller problem `milp_reduced`, and return the information needed by [`postsolve`](@ref).
"""
function presolve end

"""
    postsolve(presolver, presolve_info, sol_reduced::PrimalDualSolution) -> PrimalDualSolution

Map a solution of the reduced problem back to the original problem, keeping the element and array types of `sol_reduced`.
"""
function postsolve end

"""
    PaPILOPresolver(; verbose = false)

Presolver based on [PaPILO.jl](https://github.com/scipopt/PaPILO.jl), which requires `using PaPILO`.

It presolves the continuous relaxation and recovers both primal and dual solutions.
It throws an error when PaPILO proves the problem infeasible or unbounded.

!!! warning
    The dual solution reconstructed by PaPILO can be far from optimal on the original problem, even when the reduced problem is solved accurately, in which case `solve` returns `ALMOST_OPTIMAL`.

!!! warning
    With `verbose = false`, PaPILO is muted by redirecting the `stdout` of the whole process, which is not safe when other tasks print or presolve concurrently.
"""
struct PaPILOPresolver <: AbstractPresolver
    verbose::Bool
    PaPILOPresolver(; verbose::Bool = false) = new(verbose)
end
