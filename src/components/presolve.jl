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
"""
struct PaPILOPresolver <: AbstractPresolver
    verbose::Bool
    PaPILOPresolver(; verbose::Bool = false) = new(verbose)
end
