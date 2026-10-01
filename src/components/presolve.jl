"""
    AbstractPresolver

Supertype for pluggable presolve backends. To plug a custom presolver into [`Algorithm`](@ref)
(`presolver = MyPresolver(...)`), define a subtype and implement [`presolve`](@ref) and
[`postsolve`](@ref) for it. [`PaPILOPresolver`](@ref) is the presolver built into CoolPDLP.
"""
abstract type AbstractPresolver end

"""
    presolve(presolver::AbstractPresolver, milp::MILP) -> (milp_reduced, presolve_info)

Reduce `milp` using `presolver`. Return the (typically smaller) reduced [`MILP`](@ref) to hand
over to the algorithm, together with an opaque `presolve_info` object to later pass to
[`postsolve`](@ref) along with a solution of the reduced problem.

`milp_reduced` needs no particular element or array type: `solve` feeds it back through
`preprocess`, which preconditions on the host and then calls [`perform_conversion`](@ref)
anyway. A CPU-`Float64` problem — what an external presolver naturally produces — is fine.

`presolve_info` is produced by `presolve` and consumed by `postsolve` for the *same* presolver
type, so it can be any Julia object convenient for that backend: index maps, substitution
coefficients, the contents of some intermediate file, and so on.

Note that `presolve_info` must allow returning a postsolved `PrimalDualSolution` of the correct
type with respect to the original `MILP`. Typically, that may require storing a prototype
solution.

`solve` only ever tackles the continuous relaxation of `milp` (see [`relax`](@ref)), so a
presolver is free to ignore integrality — and should not apply an integer-specific reduction,
which would reduce a problem nobody is solving.

!!! warning
    Implementations are expected to be type-stable: `solve` runs `presolve` as one of its steps,
    like `preprocess` or `initialize`, and inherits whatever `presolve` returns. A presolver
    whose return type is not inferrable makes `solve` type-unstable too, which
    [DispatchDoctor](https://github.com/MilesCranmer/DispatchDoctor.jl) turns into an error.
"""
function presolve end

"""
    presolve(::Nothing, milp::MILP) -> (milp, nothing)

Reduce nothing at all: `presolver = nothing` is how [`Algorithm`](@ref) spells "no presolve", and
this is the identity step it stands for, so that `solve` runs the same pipeline either way.
"""
presolve(::Nothing, milp::MILP) = (milp, nothing)

"""
    postsolve(presolver::AbstractPresolver, presolve_info, sol_reduced::PrimalDualSolution) -> PrimalDualSolution

Map `sol_reduced`, a solution of the reduced problem produced by [`presolve`](@ref), back to a
solution of the original problem, using `presolve_info`.

The result must have the shape of the original problem, which `presolve_info` memorized, and the
element and array types of `sol_reduced`, which the algorithm produced: `solve` hands it straight
back to the caller without converting it any further.

Both halves of the solution must be mapped back: `solve` recomputes the KKT errors of the result
on the original problem, so a dual that is not postsolved shows up as a solution that misses the
requested tolerance, which `solve` then reports as `ALMOST_OPTIMAL` at best. Implementations that
genuinely cannot reconstruct the dual (because the underlying tool's interface is primal-only)
should fill it with `NaN` rather than `0.0`: `NaN` propagates loudly through any arithmetic that
touches it, rather than being mistaken for a real (zero) dual value.

`postsolve` must leave `presolve_info` usable: the same reduction can be asked to map several
solutions back.

!!! warning
    Like [`presolve`](@ref), implementations are expected to be type-stable.
"""
function postsolve end

"""
    postsolve(::Nothing, ::Nothing, sol_reduced::PrimalDualSolution) -> sol_reduced

Map nothing back: the counterpart of [`presolve`](@ref) on a `nothing` presolver, where the
"reduced" problem was the original one all along.
"""
postsolve(::Nothing, ::Nothing, sol_reduced::PrimalDualSolution) = sol_reduced

"""
    PaPILOPresolver(; verbose = false, dual_postsolve = true)

The [`AbstractPresolver`](@ref) built into CoolPDLP: round-trips `milp` through MPS files and
calls [PaPILO.jl](https://github.com/scipopt/PaPILO.jl)'s `presolve`/`postsolve` commands.

# Fields

$(TYPEDFIELDS)

PaPILO stops with an error when it proves the problem infeasible or unbounded, and so do
[`presolve`](@ref) and `solve`.

!!! note
    PaPILO is licensed under Apache-2.0 (unlike the MIT-licensed `CoolPDLP`), so it is only a
    weak dependency: [`presolve`](@ref)/[`postsolve`](@ref) for a `PaPILOPresolver` are defined
    by the `CoolPDLPPaPILOExt` package extension, and calling them before running
    `using PaPILO` throws a `MethodError` (with a hint pointing at the missing `using`).
"""
struct PaPILOPresolver <: AbstractPresolver
    """
    whether to let PaPILO print its own progress to `stdout`. Muting it redirects the `stdout`
    of the whole process while PaPILO runs, which is not safe when several tasks print or
    presolve concurrently: use `verbose = true` in that case
    """
    verbose::Bool
    """
    whether to recover the dual solution as well as the primal one. PaPILO only records the
    information needed for that when presolving is restricted to the reductions that support
    it, which leaves a larger reduced problem. Without it the dual comes back as `NaN`, so
    `solve` never reports `OPTIMAL`
    """
    dual_postsolve::Bool

    function PaPILOPresolver(; verbose::Bool = false, dual_postsolve::Bool = true)
        return new(verbose, dual_postsolve)
    end
end

function Base.show(io::IO, presolver::PaPILOPresolver)
    (; verbose, dual_postsolve) = presolver
    return print(io, "PaPILOPresolver(verbose=$verbose, dual_postsolve=$dual_postsolve)")
end
