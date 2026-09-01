"""
    PresolveParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PresolveParameters
    "whether to presolve the MILP with PaPILO before running the algorithm"
    enabled::Bool = false
    "whether to let PaPILO print its own progress to `stdout`"
    verbose::Bool = false
end

function Base.show(io::IO, params::PresolveParameters)
    (; enabled, verbose) = params
    return print(io, "PresolveParameters: enabled=$enabled, verbose=$verbose")
end

"""
    PresolveResult

Bookkeeping produced by [`presolve_milp`](@ref), used by [`postsolve_solution`](@ref) to map
a solution of the presolved MILP back to the original problem.

# Fields

$(TYPEDFIELDS)
"""
struct PresolveResult
    "path to the postsolve archive written by PaPILO"
    postsolve_file::String
    "variable names of the original problem (as they appear in the input MPS file)"
    var_names_orig::Vector{String}
    "variable names of the presolved problem (as they appear in the reduced MPS file)"
    var_names_reduced::Vector{String}
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
    At = milp.At isa SparseMatrixCSC ? milp.At : SparseMatrixCSC(milp.At)

    model = JuMP.Model()
    x = JuMP.@variable(model, x[1:n])
    for j in 1:n
        JuMP.set_name(x[j], var_names[j])
        isfinite(lv[j]) && JuMP.set_lower_bound(x[j], lv[j])
        isfinite(uv[j]) && JuMP.set_upper_bound(x[j], uv[j])
        int_var[j] && JuMP.set_integer(x[j])
    end

    obj = zero(JuMP.AffExpr)
    for j in 1:n
        iszero(c[j]) || JuMP.add_to_expression!(obj, c[j], x[j])
    end
    JuMP.@objective(model, Min, obj)

    for i in 1:m
        expr = zero(JuMP.AffExpr)
        for k in nzrange(At, i)
            JuMP.add_to_expression!(expr, At.nzval[k], x[At.rowval[k]])
        end
        li, ui = lc[i], uc[i]
        con = if li == ui
            JuMP.@constraint(model, expr == li)
        elseif isfinite(li) && isfinite(ui)
            JuMP.@constraint(model, li <= expr <= ui)
        elseif isfinite(li)
            JuMP.@constraint(model, expr >= li)
        elseif isfinite(ui)
            JuMP.@constraint(model, expr <= ui)
        else
            JuMP.@constraint(model, expr in MOI.Interval(-Inf, Inf))
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
    write_papilo_solution(path, x, var_names)

Write the primal vector `x` (indexed like `var_names`) to `path` in the plain-text solution
format expected by PaPILO's `postsolve` command.
"""
function write_papilo_solution(path::AbstractString, x::AbstractVector, var_names::Vector{String})
    open(path, "w") do io
        println(io, "=obj= 0")
        for (name, xi) in zip(var_names, x)
            println(io, name, " ", xi)
        end
    end
    return path
end

"""
    read_papilo_solution(path, var_names)

Parse a plain-text solution file produced by PaPILO's `postsolve` command, returning a vector
of values indexed like `var_names`. Variables absent from the file default to zero.
"""
function read_papilo_solution(path::AbstractString, var_names::Vector{String})
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
    presolve_milp(milp::MILP, params::PresolveParameters)

Run PaPILO's presolve on `milp` through a round trip of MPS files, returning
`(milp_reduced, result)`.

`milp_reduced` is the (typically smaller) problem to hand over to the algorithm, and `result`
is a [`PresolveResult`](@ref) to pass to [`postsolve_solution`](@ref) once it has been solved.

If presolve fails for any reason (e.g. the PaPILO binary errors out), a warning is emitted and
`(milp, nothing)` is returned so that the caller can fall back to solving the original problem.
"""
function presolve_milp(milp::MILP, params::PresolveParameters)
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
    postsolve_solution(result::PresolveResult, sol_reduced::PrimalDualSolution, params::PresolveParameters)

Map the primal part of `sol_reduced` (a solution of the presolved problem) back to the
original problem described by `result`, using PaPILO's postsolve mechanism.

The dual part is not reconstructed (PaPILO's file-based interface only round-trips primal
solutions) and is returned as a vector of zeros.
"""
function postsolve_solution(
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

"""
    postsolve_or_passthrough(presolve_result, sol_reduced, milp_orig, params)

Map `sol_reduced` back to the space of `milp_orig`, returning a `(x, y)` couple of plain
`Vector{Float64}`.

If `presolve_result` is `nothing` (presolve was a no-op or fell back after failing), `sol_reduced`
already lives in the original space and is returned as is (converted to plain vectors).
Otherwise the primal part is mapped back with [`postsolve_solution`](@ref) and the dual part is
set to zero, since PaPILO's file-based interface does not round-trip dual solutions.
"""
function postsolve_or_passthrough(
        presolve_result, sol_reduced::PrimalDualSolution, milp_orig::MILP, params::PresolveParameters
    )
    if isnothing(presolve_result)
        return Vector{Float64}(Array(sol_reduced.x)), Vector{Float64}(Array(sol_reduced.y))
    end
    x = postsolve_solution(presolve_result, sol_reduced, params)
    y = zeros(Float64, nbcons(milp_orig))
    return x, y
end
