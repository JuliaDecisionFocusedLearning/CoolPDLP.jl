"""
    BasicPresolver(; fixed_variables, empty_rows, empty_columns, max_passes, verbose)

A basic presolve that eliminates fixed variables and empty rows/columns.

# Reduction steps

- `fixed_variables`: eliminate variables with `lv[j] == uv[j]`.
- `empty_rows`: eliminate rows that have no non-zero entries.
- `empty_columns`: eliminate variables that do not appear in constraints.

# Keyword arguments

$(TYPEDFIELDS)
"""
struct BasicPresolver <: AbstractPresolver
    "Eliminate variables fixed by their bounds (`lv[j] == uv[j]`)"
    fixed_variables::Bool
    "Remove all-zero rows (checking feasibility first)"
    empty_rows::Bool
    "Remove variables that appear in no active constraint"
    empty_columns::Bool
    "Maximum number of presolve passes"
    max_passes::Int
    "Print per-pass reduction summary"
    verbose::Bool

    function BasicPresolver(;
            fixed_variables::Bool = true,
            empty_rows::Bool      = true,
            empty_columns::Bool   = true,
            max_passes::Int       = 5,
            verbose::Bool         = false,
        )
        return new(fixed_variables, empty_rows, empty_columns, max_passes, verbose)
    end
end

function Base.show(io::IO, p::BasicPresolver)
    active = String[]
    p.fixed_variables && push!(active, "fixed_variables")
    p.empty_rows       && push!(active, "empty_rows")
    p.empty_columns    && push!(active, "empty_columns")
    steps_str = isempty(active) ? "none" : join(active, ", ")
    return print(io, "BasicPresolver(steps: $steps_str, max_passes: $(p.max_passes))")
end

"""
Result produced by [`BasicPresolver`](@ref), used to recover the original solution.

# Fields

$(TYPEDFIELDS)
"""
struct BasicPresolveResult{T} <: AbstractPresolveResult
    "var_map[i] = original index of the i-th variable in the reduced problem"
    var_map::Vector{Int}
    "con_map[i] = original index of the i-th constraint in the reduced problem"
    con_map::Vector{Int}
    "Original indices of all variables that were eliminated (fixed or empty-column)"
    fixed_var_idx::Vector{Int}
    "Value to which each eliminated variable is set in the recovered solution"
    fixed_var_val::Vector{T}
    "Number of variables in the original problem"
    n_orig::Int
    "Number of constraints in the original problem"
    m_orig::Int
    "The MILP to pass to the solver (reduced or original)"
    milp_to_solve::MILP{T, Vector{T}, SparseMatrixCSC{T, Int}, Vector{Bool}}
end

# Build a 0×0 placeholder MILP for the infeasible/unbounded early-return paths.
# This is to ensure type stability.
function _empty_milp(::Type{T}) where T
    return MILP(;
        c       = T[],
        lv      = T[],
        uv      = T[],
        A       = spzeros(T, 0, 0),
        lc      = T[],
        uc      = T[],
        int_var = Bool[],
    )
end

"""Eliminate variables fixed by substitution."""
function _pass_fixed_vars!(
        var_keep   :: BitVector,
        fixed_idx  :: Vector{Int},
        fixed_val  :: Vector{T},
        A          :: SparseMatrixCSC{T, Int},
        lc         :: Vector{T},
        uc         :: Vector{T},
        lv         :: Vector{T},
        uv         :: Vector{T},
    ) where T
    n_fixed = 0
    for j in eachindex(var_keep)
        var_keep[j] || continue
        lv[j] == uv[j] || continue
        var_keep[j] = false
        push!(fixed_idx, j)
        push!(fixed_val, lv[j])
        val = lv[j]
        for k in nzrange(A, j)
            i       = A.rowval[k]
            contrib = A.nzval[k] * val
            lc[i]  -= contrib
            uc[i]  -= contrib
        end
        n_fixed += 1
    end
    return n_fixed
end

"""
Remove rows that have no non-zero entry among `var_keep` variables.

Returns `n_removed ≥ 0` on success, or `-1` if inconsistent bounds are found.
"""
function _pass_empty_rows!(
        con_keep :: BitVector,
        var_keep :: BitVector,
        At       :: SparseMatrixCSC{T, Int},
        lc       :: Vector{T},
        uc       :: Vector{T},
    ) where T
    n_removed = 0
    for i in eachindex(con_keep)
        con_keep[i] || continue
        row_active = false
        for k in nzrange(At, i)
            if var_keep[At.rowval[k]] && At.nzval[k] != 0
                row_active = true
                break
            end
        end
        row_active && continue
        # row i is empty
        (lc[i] > 0 || uc[i] < 0) && return -1
        con_keep[i] = false
        n_removed += 1
    end
    return n_removed
end

"""
Handle variables that appear in no active constraint.

Returns `n_fixed ≥ 0` on success, or `-1` if an unbounded variable is found.
"""
function _pass_empty_cols!(
        var_keep  :: BitVector,
        fixed_idx :: Vector{Int},
        fixed_val :: Vector{T},
        A         :: SparseMatrixCSC{T, Int},
        con_keep  :: BitVector,
        lv        :: Vector{T},
        uv        :: Vector{T},
        c         :: Vector{T},
    ) where T
    n_fixed = 0
    for j in eachindex(var_keep)
        var_keep[j] || continue
        col_active = false
        for k in nzrange(A, j)
            if con_keep[A.rowval[k]] && A.nzval[k] != 0
                col_active = true
                break
            end
        end
        col_active && continue
        # column j is empty
        cj = c[j]
        if cj > 0  # lower (or unbounded)
            lv[j] == -Inf && return -1
            push!(fixed_idx, j); push!(fixed_val, lv[j])
        elseif cj < 0  # upper (or unbounded)
            uv[j] == Inf && return -1
            push!(fixed_idx, j); push!(fixed_val, uv[j])
        else  # any value between lower and upper
            push!(fixed_idx, j); push!(fixed_val, clamp(zero(T), lv[j], uv[j]))
        end
        var_keep[j] = false
        n_fixed += 1
    end
    return n_fixed
end

function apply_presolve(presolver::BasicPresolver, milp::MILP{T}) where T
    n = nbvar(milp)
    m = nbcons(milp)

    (; lv, uv, lc, uc, c, int_var, A, At) = milp
    var_names = copy(milp.var_names)

    var_keep  = trues(n)
    con_keep  = trues(m)
    fixed_idx = Int[]
    fixed_val = T[]

    total_fixed   = 0
    total_removed = 0

    for pass in 1:presolver.max_passes
        pass_fixed   = 0
        pass_removed = 0

        if presolver.fixed_variables
            pass_fixed += _pass_fixed_vars!(
                var_keep, fixed_idx, fixed_val, A, lc, uc, lv, uv)
        end

        if presolver.empty_rows
            r = _pass_empty_rows!(con_keep, var_keep, At, lc, uc)
            if r == -1
                empty = _empty_milp(T)
                return PresolveInfeasible,
                    BasicPresolveResult(Int[], Int[], Int[], T[], n, m, empty)
            end
            pass_removed += r
        end

        if presolver.empty_columns
            r = _pass_empty_cols!(
                var_keep, fixed_idx, fixed_val, A, con_keep, lv, uv, c)
            if r == -1
                empty = _empty_milp(T)
                return PresolveUnbounded,
                    BasicPresolveResult(Int[], Int[], Int[], T[], n, m, empty)
            end
            pass_fixed += r
        end

        total_fixed   += pass_fixed
        total_removed += pass_removed

        if presolver.verbose
            (pass_fixed > 0 || pass_removed > 0) &&
                @info "BasicPresolver pass $pass: \
                       fixed $pass_fixed variable(s), \
                       removed $pass_removed constraint(s)"
        end

        pass_fixed == 0 && pass_removed == 0 && break
    end

    var_map = findall(var_keep)   # reduced_var_idx  →  orig_var_idx
    con_map = findall(con_keep)   # reduced_con_idx  →  orig_con_idx

    # FIXME: don't rebuild if unchanged
    milp_red = MILP(;
        c         = c[var_map],
        lv        = lv[var_map],
        uv        = uv[var_map],
        A         = A[con_map, var_map],
        lc        = lc[con_map],
        uc        = uc[con_map],
        int_var   = int_var[var_map],
        var_names = var_names[var_map],
        dataset   = milp.dataset,
        name      = milp.name,
        path      = milp.path,
    )

    result = BasicPresolveResult(var_map, con_map, fixed_idx, fixed_val, n, m, milp_red)

    if total_fixed == 0 && total_removed == 0
        return PresolveUnchanged, result
    else
        return PresolveReduced, result
    end
end

function map_warmstart(res::BasicPresolveResult{T}, sol::PrimalDualSolution{T}) where T
    return PrimalDualSolution(sol.x[res.var_map], sol.y[res.con_map])
end

function recover_solution(res::BasicPresolveResult{T}, reduced_sol::PrimalDualSolution{T}) where T
    x_orig = zeros(T, res.n_orig)
    y_orig = zeros(T, res.m_orig)

    x_orig[res.var_map] .= reduced_sol.x
    y_orig[res.con_map] .= reduced_sol.y

    for (j, v) in zip(res.fixed_var_idx, res.fixed_var_val)
        x_orig[j] = T(v)
    end

    return PrimalDualSolution(x_orig, y_orig)
end
