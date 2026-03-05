module PSLPExt

using CoolPDLP: CoolPDLP, MILP, PrimalDualSolution, AbstractPresolver, AbstractPresolveResult,
    PresolveUnchanged, PresolveReduced, PresolveInfeasible, PresolveUnboundedOrInfeasible
import PSLP
using SparseArrays: SparseMatrixCSC, sparse, spzeros

function __init__()
    setglobal!(CoolPDLP, :PSLPPresolver, PSLPPresolver)
    return
end

"""
    PSLPPresolver(; verbose=false)

PSLP-based presolver.  Requires loading the `PSLP` package.
"""
struct PSLPPresolver <: AbstractPresolver
    verbose::Bool
    PSLPPresolver(; verbose::Bool = false) = new(verbose)
end

struct PSLPPresolveResult <: AbstractPresolveResult
    handle::Union{Nothing, PSLP.API.Handle}  # FIXME: empty result instead of nothing
    milp_to_solve::MILP{Float64, Vector{Float64}, SparseMatrixCSC{Float64, Int}, Vector{Bool}}
end

function _empty_milp()
    return MILP(;
        c = Float64[], lv = Float64[], uv = Float64[],
        A = spzeros(Float64, 0, 0), lc = Float64[], uc = Float64[], int_var = Bool[],
    )
end

_as_pslp_milp(milp::MILP{Float64, Vector{Float64}, SparseMatrixCSC{Float64, Int}, Vector{Bool}}) = milp
function _as_pslp_milp(milp::MILP)
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

function _pslp_status(s::PSLP.API.Status)
    s == PSLP.API.Unchanged  && return PresolveUnchanged
    s == PSLP.API.Reduced    && return PresolveReduced
    s == PSLP.API.Infeasible && return PresolveInfeasible
    return PresolveUnboundedOrInfeasible
end

function CoolPDLP.apply_presolve(ps::PSLPPresolver, milp::MILP)
    (; c, lv, uv, At, lc, uc) = milp

    At_csc = At isa SparseMatrixCSC ? At : SparseMatrixCSC(At)
    Ap = Int32.(At_csc.colptr .- 1)
    Ai = Int32.(At_csc.rowval .- 1)
    Ax = At_csc.nzval
    m  = size(milp.A, 1)
    n  = size(milp.A, 2)

    handle = PSLP.API.Handle(
        Ax, Ai, Ap, m, n, length(Ax),
        lc, uc,
        lv, uv,
        c;
        verbose = ps.verbose,
    )

    st = _pslp_status(PSLP.API.run(handle))

    if st == PresolveUnchanged
        return PresolveUnchanged, PSLPPresolveResult(nothing, _as_pslp_milp(milp))
    elseif st != PresolveReduced
        return st, PSLPPresolveResult(nothing, _empty_milp())
    end

    m_red   = PSLP.API.red_m(handle)
    n_red   = PSLP.API.red_n(handle)
    Ap_red, Ai_red, Ax_red = PSLP.API.get_matrix(handle)
    lc_red, uc_red = PSLP.API.get_row_bounds(handle)
    lv_red, uv_red = PSLP.API.get_col_bounds(handle)
    c_red          = PSLP.API.get_obj(handle)

    row_inds = [i for i in 1:m_red for _ in (Ap_red[i] + 1):Ap_red[i + 1]]
    A_red    = sparse(row_inds, Ai_red .+ 1, Ax_red, m_red, n_red)

    milp_red = MILP(;
        c  = c_red,
        lv = lv_red, uv = uv_red,
        A  = A_red,
        lc = lc_red, uc = uc_red,
    )

    return PresolveReduced, PSLPPresolveResult(handle, milp_red)
end

function CoolPDLP.map_warmstart(r::PSLPPresolveResult, sol::PrimalDualSolution)
    r.handle === nothing && return sol

    x_red = PSLP.API.map_primal(r.handle, sol.x)

    row_map = PSLP.API.get_row_map(r.handle)
    y_red = zeros(Float64, PSLP.API.red_m(r.handle))
    for (j, i) in enumerate(row_map)
        i > 0 && (y_red[i] = sol.y[j])
    end

    return PrimalDualSolution(x_red, y_red)
end

function CoolPDLP.recover_solution(r::PSLPPresolveResult, reduced_sol::PrimalDualSolution)
    r.handle === nothing && return reduced_sol

    x_red = reduced_sol.x
    y_red = reduced_sol.y

    z_red = r.milp_to_solve.c - r.milp_to_solve.At * y_red
    PSLP.API.postsolve!(r.handle, x_red, y_red, z_red)

    x_orig = PSLP.API.get_x(r.handle)
    y_orig = PSLP.API.get_y(r.handle)

    return PrimalDualSolution(x_orig, y_orig)
end

end # module PSLPExt
