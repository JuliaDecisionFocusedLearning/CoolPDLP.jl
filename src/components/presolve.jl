"""
    AbstractPresolver

Supertype for pluggable presolve backends. To plug a custom presolver into [`Algorithm`](@ref)
(`presolve = MyPresolver(...)`), define a subtype and implement [`presolve`](@ref) and
[`postsolve`](@ref) for it. [`PaPILOPresolver`](@ref) is the presolver built into CoolPDLP.
"""
abstract type AbstractPresolver end

"""
    presolve(presolver::AbstractPresolver, milp::MILP) -> (milp_reduced, state)

Reduce `milp` using `presolver`. Return the (typically smaller) reduced [`MILP`](@ref) to hand
over to the algorithm, together with an opaque `state` object to later pass to
[`postsolve`](@ref) along with a solution of the reduced problem.

`state` is produced by `presolve` and consumed by `postsolve` for the *same* presolver type, so
it can be any Julia object convenient for that backend: index maps, substitution coefficients,
a path to some intermediate file, ... there is no file-based or otherwise constrained contract
here, unlike [`PaPILOPresolver`](@ref)'s own state which happens to hold a file path because
that particular backend is file-based.
"""
function presolve end

"""
    postsolve(presolver::AbstractPresolver, state, sol_reduced::PrimalDualSolution) -> PrimalDualSolution

Map `sol_reduced`, a solution of the reduced problem produced by [`presolve`](@ref), back to a
solution of the original problem, using `state`.

Implementations that cannot reconstruct the dual solution (e.g. because the underlying tool's
interface is primal-only, like [`PaPILOPresolver`](@ref)'s) should fill it with `NaN` rather
than `0.0`: `NaN` propagates loudly through any arithmetic that touches it, rather than being
mistaken for a real (zero) dual value.
"""
function postsolve end

"""
    PresolveParameters{P}

`P`, the type of the configured presolver (`Nothing` when presolve is disabled), is a type
parameter rather than a field, much like [`isbatched`](@ref) for a [`MILP`](@ref): this lets
`solve` dispatch on it at compile time, so that solving without presolve never needs to compile
the presolve code path at all.

# Fields

$(TYPEDFIELDS)
"""
struct PresolveParameters{P <: Union{Nothing, AbstractPresolver}}
    "the presolver to use, or `nothing` to disable presolve"
    presolver::P
    "whether to let a presolve failure error instead of falling back to the original problem"
    strict::Bool

    function PresolveParameters(;
            presolver::Union{Nothing, AbstractPresolver} = nothing, strict::Bool = false
        )
        return new{typeof(presolver)}(presolver, strict)
    end
end

"""
    presolve_enabled(params::PresolveParameters)

Return whether presolve is enabled, as a plain `Bool` extracted from the type of `params`.
"""
presolve_enabled(::PresolveParameters{Nothing}) = false
presolve_enabled(::PresolveParameters) = true

function Base.show(io::IO, params::PresolveParameters)
    (; presolver, strict) = params
    return print(io, "PresolveParameters: presolver=$presolver, strict=$strict")
end

"""
    milp_to_mps(milp::MILP, path::AbstractString)

Write `milp` to an MPS file at `path`, using a [JuMP](https://github.com/jump-dev/JuMP.jl)
model as an intermediate representation.

Every variable bound is set explicitly (even when infinite), so that reading the file back
does not depend on the default bounds assumed by the MPS format.
"""
function milp_to_mps(milp::MILP, path::AbstractString)
    (; c, lv, uv, lc, uc, int_var, var_names) = milp
    n, m = nbvar(milp), nbcons(milp)
    A = milp.A isa SparseMatrixCSC ? milp.A : SparseMatrixCSC(milp.A)

    model = JuMP.Model()
    x = JuMP.@variable(model, x[1:n])
    JuMP.set_name.(x, var_names)
    finite_lv, finite_uv = isfinite.(lv), isfinite.(uv)
    JuMP.set_lower_bound.(x[finite_lv], lv[finite_lv])
    JuMP.set_upper_bound.(x[finite_uv], uv[finite_uv])
    JuMP.set_integer.(x[int_var])

    JuMP.@objective(model, Min, dot(c, x))

    Ax = A * x
    for i in 1:m
        li, ui = lc[i], uc[i]
        con = if li == ui
            JuMP.@constraint(model, Ax[i] == li)
        elseif isfinite(li) && isfinite(ui)
            JuMP.@constraint(model, li <= Ax[i] <= ui)
        elseif isfinite(li)
            JuMP.@constraint(model, Ax[i] >= li)
        elseif isfinite(ui)
            JuMP.@constraint(model, Ax[i] <= ui)
        else
            JuMP.@constraint(model, Ax[i] in MOI.Interval(-Inf, Inf))
        end
        JuMP.set_name(con, "R$i")
    end

    JuMP.write_to_file(model, path; format = MOI.FileFormats.FORMAT_MPS)
    return path
end

_setbounds(s::MOI.EqualTo) = (s.value, s.value)
_setbounds(s::MOI.LessThan) = (-Inf, s.upper)
_setbounds(s::MOI.GreaterThan) = (s.lower, Inf)
_setbounds(s::MOI.Interval) = (s.lower, s.upper)

"""
    mps_to_milp(path::AbstractString; kwargs...)

Read the MPS file at `path` into a [`MILP`](@ref), using a
[JuMP](https://github.com/jump-dev/JuMP.jl) model as an intermediate representation.

`kwargs` are forwarded to the [`MILP`](@ref) constructor.
"""
function mps_to_milp(path::AbstractString; kwargs...)
    model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    vars = JuMP.all_variables(model)
    n = length(vars)
    col = Dict(v => j for (j, v) in enumerate(vars))
    var_names = JuMP.name.(vars)

    lv = fill(-Inf, n)
    uv = fill(Inf, n)
    int_var = zeros(Bool, n)
    for (j, v) in enumerate(vars)
        if JuMP.is_fixed(v)
            lv[j] = uv[j] = JuMP.fix_value(v)
        else
            JuMP.has_lower_bound(v) && (lv[j] = JuMP.lower_bound(v))
            JuMP.has_upper_bound(v) && (uv[j] = JuMP.upper_bound(v))
        end
        (JuMP.is_binary(v) || JuMP.is_integer(v)) && (int_var[j] = true)
        if JuMP.is_binary(v)
            lv[j] = max(lv[j], 0.0)
            uv[j] = min(uv[j], 1.0)
        end
    end

    c = zeros(n)
    obj = JuMP.objective_function(model, JuMP.AffExpr)
    for (v, coeff) in obj.terms
        c[col[v]] += coeff
    end
    JuMP.objective_sense(model) == MOI.MAX_SENSE && (c .*= -1)

    rows_i, rows_j, rows_v = Int[], Int[], Float64[]
    lc, uc = Float64[], Float64[]
    row = 0
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: JuMP.AffExpr || continue
        for cref in JuMP.all_constraints(model, F, S)
            row += 1
            cobj = JuMP.constraint_object(cref)
            for (v, coeff) in cobj.func.terms
                push!(rows_i, row)
                push!(rows_j, col[v])
                push!(rows_v, coeff)
            end
            li, ui = _setbounds(cobj.set)
            push!(lc, li)
            push!(uc, ui)
        end
    end
    m = row
    A = sparse(rows_i, rows_j, rows_v, m, n)
    At = sparse(rows_j, rows_i, rows_v, n, m)

    return MILP(; c, lv, uv, A, At, lc, uc, int_var, var_names, kwargs...)
end

"""
    write_sol_file(path, x, var_names)

Write the primal vector `x` (indexed like `var_names`) to `path` in the plain-text `.sol`
solution format shared by SCIP, PaPILO and several other solvers in the SCIP ecosystem: a
header line, then one `name value` line per variable.
"""
function write_sol_file(path::AbstractString, x::AbstractVector, var_names::Vector{String})
    open(path, "w") do io
        println(io, "=obj= 0")
        for (name, xi) in zip(var_names, x)
            println(io, name, " ", xi)
        end
    end
    return path
end

"""
    read_sol_file(path, var_names)

Parse a plain-text `.sol` file (the format shared by SCIP, PaPILO and several other solvers in
the SCIP ecosystem), returning a vector of values indexed like `var_names`. Variables absent
from the file default to zero.
"""
function read_sol_file(path::AbstractString, var_names::Vector{String})
    x = zeros(length(var_names))
    idx = Dict(name => j for (j, name) in enumerate(var_names))
    for line in eachline(path)
        startswith(line, "=obj=") && continue
        tokens = split(line)
        isempty(tokens) && continue
        j = get(idx, tokens[1], nothing)
        isnothing(j) && continue
        x[j] = parse(Float64, tokens[2])
    end
    return x
end

"""
    PaPILOPresolver(; verbose = false)

The [`AbstractPresolver`](@ref) built into CoolPDLP: round-trips `milp` through MPS files and
calls [PaPILO.jl](https://github.com/scipopt/PaPILO.jl)'s `presolve`/`postsolve` commands.

# Fields

$(TYPEDFIELDS)

!!! note
    PaPILO is licensed under Apache-2.0 (unlike the MIT-licensed `CoolPDLP`), so it is only a
    weak dependency: [`presolve`](@ref)/[`postsolve`](@ref) on a `PaPILOPresolver` are
    implemented by the `CoolPDLPPaPILOExt` package extension, and calling them without having
    run `using PaPILO` first throws an informative error.
"""
struct PaPILOPresolver <: AbstractPresolver
    "whether to let PaPILO print its own progress to `stdout`"
    verbose::Bool

    PaPILOPresolver(; verbose::Bool = false) = new(verbose)
end

function Base.show(io::IO, presolver::PaPILOPresolver)
    return print(io, "PaPILOPresolver(verbose=$(presolver.verbose))")
end

"""
    PaPILOPresolveState

The `state` object produced by `presolve(::PaPILOPresolver, milp)` and consumed by
`postsolve(::PaPILOPresolver, state, sol_reduced)`.

# Fields

$(TYPEDFIELDS)
"""
struct PaPILOPresolveState
    "path to the postsolve archive written by PaPILO"
    postsolve_file::String
    "variable names of the original problem (as they appear in the input MPS file)"
    var_names_orig::Vector{String}
    "variable names of the presolved problem (as they appear in the reduced MPS file)"
    var_names_reduced::Vector{String}
    "number of constraints in the original problem (the dual, not reconstructed, is `NaN`-filled at this length)"
    nbcons_orig::Int
end

function presolve(presolver::PaPILOPresolver, milp::MILP)
    ext = Base.get_extension(@__MODULE__, :CoolPDLPPaPILOExt)
    isnothing(ext) && _error_papilo_not_loaded()
    return ext.papilo_presolve(presolver, milp)
end

function postsolve(presolver::PaPILOPresolver, state::PaPILOPresolveState, sol_reduced::PrimalDualSolution)
    ext = Base.get_extension(@__MODULE__, :CoolPDLPPaPILOExt)
    isnothing(ext) && _error_papilo_not_loaded()
    return ext.papilo_postsolve(presolver, state, sol_reduced)
end

function _error_papilo_not_loaded()
    return error(
        "Presolve requires PaPILO.jl to be loaded first (it is a weak dependency of " *
            "CoolPDLP, kept optional because of its Apache-2.0 license): run `using PaPILO` " *
            "and try again."
    )
end
