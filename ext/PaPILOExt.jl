module PaPILOExt

# FIXME: use DispatchDoctor
using CoolPDLP: CoolPDLP, MILP, PrimalDualSolution, AbstractPresolver, AbstractPresolveResult,
    PresolveUnchanged, PresolveReduced, PresolveInfeasible, PresolveUnbounded,
    PresolveUnboundedOrInfeasible
using PaPILO: API
using SparseArrays: SparseMatrixCSC, sparse, spzeros

function __init__()
    setglobal!(CoolPDLP, :PaPILOPresolver, PaPILOPresolver)
    return
end

"""
    PaPILOPresolver(; threads=1, verbosity=0)

PaPILO-based presolver.  Requires loading the `PaPILO` package.
"""
struct PaPILOPresolver <: AbstractPresolver
    threads::Int
    verbosity::Int
    PaPILOPresolver(; threads::Int = 1, verbosity::Int = 0) = new(threads, verbosity)
end

struct PaPILOPresolveResult <: AbstractPresolveResult
    result::Union{Nothing, API.Result}  # FIXME: empty result instead of nothing
    milp_to_solve::MILP{Float64, Vector{Float64}, SparseMatrixCSC{Float64, Int}, Vector{Bool}}
end

function _empty_milp()
    return MILP(;
        c = Float64[], lv = Float64[], uv = Float64[],
        A = spzeros(Float64, 0, 0), lc = Float64[], uc = Float64[], int_var = Bool[],
    )
end
_as_papilo_milp(milp::MILP{Float64, Vector{Float64}, SparseMatrixCSC{Float64, Int}, Vector{Bool}}) = milp
function _as_papilo_milp(milp)
    return MILP(;
        c         = Vector{Float64}(milp.c),
        lv        = Vector{Float64}(milp.lv),
        uv        = Vector{Float64}(milp.uv),
        A         = SparseMatrixCSC{Float64, Int}(milp.A),
        lc        = Vector{Float64}(milp.lc),
        uc        = Vector{Float64}(milp.uc),
        int_var   = Vector{Bool}(milp.int_var),
        var_names = copy(milp.var_names),
        dataset   = milp.dataset,
        name      = milp.name,
        path      = milp.path,
    )
end

function _api_to_presolve_status(s::API.Status)
    s == API.Unchanged  && return PresolveUnchanged
    s == API.Reduced    && return PresolveReduced
    s == API.Infeasible && return PresolveInfeasible
    s == API.Unbounded  && return PresolveUnbounded
    return PresolveUnboundedOrInfeasible
end

function CoolPDLP.apply_presolve(p::PaPILOPresolver, milp::MILP)
    (; c, lv, uv, A, At, lc, uc) = milp

    At_csc = At isa SparseMatrixCSC ? At : SparseMatrixCSC(At)
    api_result = API.apply(
        API.Presolver(; p.threads, p.verbosity),
        length(c), size(A, 1),
        c, 0.0,
        lv, uv, lc, uc,
        At_csc.colptr, At_csc.rowval, At_csc.nzval,
    )

    status = _api_to_presolve_status(API.status(api_result))

    if status == PresolveUnchanged
        return PresolveUnchanged, PaPILOPresolveResult(nothing, _as_papilo_milp(milp))
    elseif status != PresolveReduced
        return status, PaPILOPresolveResult(nothing, _empty_milp())
    end

    c_red, _       = API.get_obj(api_result)
    lv_red, uv_red = API.get_col_bounds(api_result)
    lc_red, uc_red = API.get_row_bounds(api_result)
    rs, ci, nzv    = API.get_matrix(api_result)

    ncols_red = API.num_cols(api_result)
    nrows_red = API.num_rows(api_result)
    row_inds  = [i for i in 1:nrows_red for _ in rs[i]:(rs[i + 1] - 1)]
    A_red     = sparse(row_inds, ci, nzv, nrows_red, ncols_red)

    milp_red = MILP(; c = c_red, lv = lv_red, uv = uv_red, A = A_red,
                      lc = lc_red, uc = uc_red)
    return PresolveReduced, PaPILOPresolveResult(api_result, milp_red)
end

function CoolPDLP.map_warmstart(r::PaPILOPresolveResult, sol::PrimalDualSolution)
    r.result === nothing && return sol

    x_red = API.map_primal(r.result, Vector{Float64}(sol.x))

    row_map = API.get_row_map(r.result)
    y_red = zeros(Float64, API.num_rows(r.result))
    for (i, j) in enumerate(row_map)
        j > 0 && (y_red[i] = sol.y[j])
    end

    return PrimalDualSolution(x_red, y_red)
end

function CoolPDLP.recover_solution(r::PaPILOPresolveResult, reduced_sol::PrimalDualSolution)
    r.result === nothing && return reduced_sol

    x_orig  = API.postsolve(r.result, Vector{Float64}(reduced_sol.x))
    row_map = API.get_row_map(r.result)

    y_orig = zeros(Float64, API.orig_rows(r.result))
    for (i, j) in enumerate(row_map)
        j > 0 && (y_orig[j] = reduced_sol.y[i])
    end

    return PrimalDualSolution(x_orig, y_orig)
end

end # module PaPILOExt
