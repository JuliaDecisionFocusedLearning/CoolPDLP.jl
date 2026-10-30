# Basis completion, cleanup, KKT (Algo 2 finalize; host LinearAlgebra).

"""Demote column `j` to a *finite* bound (never snap to ±Inf)."""
function _demote!(st::_SpiralState, j::Int)
    st.basis[j] = _bound_side(st.z[j], st.lv[j], st.uv[j])
    return st
end

"""
One column-pivoted QR, size `m`.

Basics first. If they are short of rank `m`, dual-active columns are added
and the same QR is run once on that list.

A free column has no finite bound, so it cannot be pinned and it stays
basic. If the QR dropped it, a bounded column of the kept set is demoted
in its place. Example: `x₁` free and `x₂ ≥ 0` in a rank-1 row. The QR
keeps `x₂`. `x₁` is swapped back in, and `x₂` goes to its bound.
"""
function _linear_independence!(st::_SpiralState)
    m = size(st.A, 1)
    _update_reduced_costs!(st)
    basics = findall(==(_BASIC), st.basis)

    function _commit!(keep_idx::Vector{Int})
        keepset = Set(keep_idx)
        # A free column has no finite bound, so it stays basic.
        for j in basics
            free = !(isfinite(st.lv[j]) || isfinite(st.uv[j]))
            if free && !(j in keepset)
                for k in keepset
                    if isfinite(st.lv[k]) || isfinite(st.uv[k])
                        delete!(keepset, k)
                        push!(keepset, j)
                        break
                    end
                end
            end
        end
        for j in basics
            j in keepset || _demote!(st, j)
        end
        for j in keepset
            st.basis[j] = _BASIC
        end
        return _recount_basics!(st).n_basis == m
    end

    cand = Int[]
    try
        if !isempty(basics)
            keep, _ = qr_keep(basics, st.A, m)
            length(keep) >= m && return _commit!(keep[1:m])
            cand = keep
        end
        extra = findall(j -> st.dual_active[j] && st.basis[j] != _BASIC, eachindex(st.basis))
        append!(cand, extra)
        length(cand) < m && return false
        kept, _ = qr_keep(cand, st.A, m)
        length(kept) == m || return false
        return _commit!(kept)
    catch
        return false
    end
end

"""`A_B x_B = b − A_N x_N`, then `A_Bᵀ y = c_B`, by a dense `\\`."""
function _solve_basic!(st::_SpiralState)
    m = size(st.A, 1)
    basic = findall(==(_BASIC), st.basis)
    length(basic) == m || return false
    T = eltype(st.x)
    AB = Matrix{T}(st.A[:, basic])
    try
        xb = AB \ Vector{T}(st.b_work)
        yb = transpose(AB) \ Vector{T}(st.c[basic])
        (any(!isfinite, xb) || any(!isfinite, yb)) && return false
        st.x[basic] = xb
        st.y = yb
    catch
        return false
    end
    return true
end

"""Square solve, then project basic columns onto their bounds."""
function _cleanup!(st::_SpiralState)
    st.n_basis == size(st.A, 1) || return false
    _solve_basic!(st) || return false
    for j in eachindex(st.basis)
        st.basis[j] == _BASIC && (st.x[j] = clamp(st.x[j], st.lv[j], st.uv[j]))
    end
    _update_reduced_costs!(st)
    return true
end

"""Relative primal + dual-sign residual on the lift (`b = 0` ⇒ ‖Ax‖).

Same shape of scale as [`kkt_errors!`](@ref) (`1 + ‖·‖`), compared with the
algorithm `termination_reltol`. It does not call `kkt_errors!`: the lift is
an equality form, so the primal residual is `‖Ax‖` and the dual residual is
the reduced-cost sign violation.
"""
function _kkt_ok(st::_SpiralState)
    tol = st.reltol
    prim = norm(st.A * st.x) / (1 + norm(st.x))
    _update_reduced_costs!(st)
    dual = zero(eltype(st.z))
    for j in eachindex(st.z)
        l, u, zj = st.lv[j], st.uv[j], st.z[j]
        z0 = zero(zj)
        if !isfinite(l) && !isfinite(u)
            dual = max(dual, abs(zj))
        elseif !isfinite(l)
            dual = max(dual, max(zj, z0))
        elseif !isfinite(u)
            dual = max(dual, max(-zj, z0))
        elseif st.basis[j] == _AT_LB
            dual = max(dual, max(-zj, z0))
        elseif st.basis[j] == _AT_UB
            dual = max(dual, max(zj, z0))
        end
    end
    return prim <= tol && dual / (1 + norm(st.c)) <= tol
end

"""Near-bound snap on structural variables `1:n`, at the algorithm `zero_tol`."""
function _snap_structural!(st::_SpiralState, n::Int)
    tol = st.zero_tol
    for j in 1:n
        l, u, xj = st.lv[j], st.uv[j], st.x[j]
        isfinite(l) && abs(xj - l) <= tol && (st.x[j] = l)
        isfinite(u) && abs(xj - u) <= tol && (st.x[j] = u)
    end
    return st
end

"""Fresh lift. Every ray stops at the same absolute `deadline`."""
function _new_spiral_state(milp, sol, zero_tol, reltol; t0, limit)
    st = _lift_boxed(milp, sol.x, sol.y)
    st.zero_tol = eltype(st.x)(zero_tol)
    st.reltol = eltype(st.x)(reltol)
    st.deadline = t0 + limit
    _project_bounds!(st)
    return st
end

function _spiral_pushes!(st::_SpiralState, cx::SpiralCrossover, rng, m::Int)
    _primal_push!(st, cx, rng)
    _time_up(st) && return st
    st.n_basis < m ? _dual_push!(st, cx, rng) : _cal_dual_active!(st)
    return st
end

"""One primal/dual push."""
function _push_once!(milp, sol, cx, zero_tol, reltol; t0, limit)
    st = _new_spiral_state(milp, sol, zero_tol, reltol; t0, limit)
    rng = cx.seed >= 0 ? StableRNG(cx.seed) : Random.default_rng()
    _spiral_pushes!(st, cx, rng, size(st.A, 1))
    return st
end

"""
    crossover!(sol, milp, cx::SpiralCrossover; zero_tol=1e-8, reltol=1e-4, time_limit=Inf)

Batched MILP: serial per [`instance`](@ref). `time_limit` is the budget left
on the solve clock. Every ray shares it. `zero_tol` and `reltol` are the
algorithm `zero_tol` and `termination_reltol`; the defaults match `PDLP()`.
A finite `(x, y)` is written back even when the primal residual grew.
A non-finite value, or an exception, keeps the PDLP point.
"""
function crossover!(
        sol::PrimalDualSolution, milp::MILP, cx::SpiralCrossover;
        zero_tol::Real = 1.0e-8,
        reltol::Real = 1.0e-4,
        time_limit::Real = Inf,
    )
    if isbatched(milp)
        for i in 1:nbinstances(milp)
            crossover!(instance(sol, i), instance(milp, i), cx; zero_tol, reltol, time_limit)
        end
        return sol
    end
    n, m = nbvar(milp), nbcons(milp)
    m == 0 && return sol

    t0, limit = time(), Float64(time_limit)
    try
        st = _push_once!(milp, sol, cx, zero_tol, reltol; t0, limit)

        # Skip the square cleanup when the wall budget is already gone
        # (it densifies A_B and has no per-call timeout).
        if !_time_up(st) && _linear_independence!(st)
            _improve_primal!(st)
            !_kkt_ok(st) && _cleanup!(st)
            _snap_structural!(st, n)
        end
        if any(!isfinite, view(st.x, 1:n)) || any(!isfinite, st.y)
            @warn "SpiralCrossover produced non-finite values; keeping PDLP point"
            return sol
        end
        copyto!(sol.x, adapt(get_backend(sol.x), st.x[1:n]))
        copyto!(sol.y, adapt(get_backend(sol.y), st.y))
    catch err
        @warn "SpiralCrossover failed; keeping PDLP point" exception = (err, catch_backtrace())
        return sol
    end
    return sol
end
