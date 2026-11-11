# Basis completion and cleanup (Algo 2 finalize; host LinearAlgebra).

"""
Columns to add so `basics` reaches `m` independent columns.

`basics` is kept as a block. The QR runs on the orthogonal complement, the
same role as the reference Schur step: a global pivot on `vcat(basics, extra)`
can demote a column the pushes just left basic.
"""
function _extend_basics(basics::Vector{Int}, extra::Vector{Int}, A::AbstractMatrix, m::Int)
    isempty(extra) && return Int[]
    if isempty(basics)
        kept, _ = qr_keep(extra, A, m)
        return length(kept) == m ? Vector{Int}(kept) : Int[]
    end
    T = eltype(A)
    AB = Matrix{T}(A[:, basics])
    FB = qr(AB, ColumnNorm())
    r = qr_numrank(FB)
    (r < length(basics) || r >= m) && return Int[]
    QAE = FB.Q' * Matrix{T}(A[:, extra])
    k = m - r
    Fcomp = qr(QAE[(r + 1):end, :], ColumnNorm())
    qr_numrank(Fcomp) < k && return Int[]
    return vcat(basics, extra[Fcomp.p[1:k]])
end

"""Demote column `j` to a *finite* bound (never snap to ±Inf)."""
function _demote!(st::_SpiralState, j::Int)
    st.basis[j] = _bound_side(st.z[j], st.lv[j], st.uv[j])
    return st
end

"""
Rank of the columns the pushes left basic. No column of that set is dropped.

More than `m` basics, or a rank-deficient set: leave the push point. The
replacement basis would pin an interior column, and its basic solution can
leave the bounds. Fewer than `m`: the current basics stay, and one
column-pivoted QR on their orthogonal complement may add dual-active columns.

A free column has no finite bound, so it cannot be pinned. If a completion
dropped it, a bounded column of the kept set is demoted in its place.
"""
function _linear_independence!(st::_SpiralState)
    m = size(st.A, 1)
    _update_reduced_costs!(st)
    basics = findall(==(_BASIC), st.basis)
    # Thinning an oversized basic set is what makes `x_B` leave the box.
    length(basics) > m && return false

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

    try
        if !isempty(basics)
            keep, _ = qr_keep(basics, st.A, m)
            # Replacing a dependent basic set would pin one of its columns.
            length(keep) < length(basics) && return false
            length(basics) == m && return _commit!(basics)
        end
        extra = findall(j -> st.dual_active[j] && st.basis[j] != _BASIC, eachindex(st.basis))
        kept = _extend_basics(basics, extra, st.A, m)
        length(kept) == m || return false
        return _commit!(kept)
    catch
        return false
    end
end

"""`A_B x_B = b − A_N x_N`, then `A_Bᵀ y = c_B`, by a dense `\\`.

`x_B` is written only when it already lies within the bounds, up to
`zero_tol`. Projecting a basic solution from outside the box leaves the
nullspace of the lift.
"""
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
        ε = st.zero_tol
        for (k, j) in enumerate(basic)
            xj, l, u = xb[k], st.lv[j], st.uv[j]
            isfinite(l) && xj < l - ε && return false
            isfinite(u) && xj > u + ε && return false
        end
        st.x[basic] = xb
        st.y = yb
    catch
        return false
    end
    return true
end

"""Square solve. Basic columns already within `zero_tol` of a bound are snapped."""
function _cleanup!(st::_SpiralState)
    st.n_basis == size(st.A, 1) || return false
    _solve_basic!(st) || return false
    for j in eachindex(st.basis)
        st.basis[j] == _BASIC && (st.x[j] = clamp(st.x[j], st.lv[j], st.uv[j]))
    end
    _update_reduced_costs!(st)
    return true
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
function _new_spiral_state(milp, sol, zero_tol; t0, limit)
    st = _lift_boxed(milp, sol.x, sol.y)
    st.zero_tol = eltype(st.x)(zero_tol)
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
function _push_once!(milp, sol, cx, zero_tol; t0, limit)
    st = _new_spiral_state(milp, sol, zero_tol; t0, limit)
    rng = cx.seed >= 0 ? StableRNG(cx.seed) : Random.default_rng()
    _spiral_pushes!(st, cx, rng, size(st.A, 1))
    return st
end

"""
    crossover!(sol, milp, cx::SpiralCrossover; zero_tol=1e-8, reltol=1e-4, time_limit=Inf)

Batched MILP: serial per [`instance`](@ref). `time_limit` is one budget for
the whole batch: a later instance only sees what is left. `zero_tol` and
`reltol` are the algorithm `zero_tol` and `termination_reltol`; the defaults
match `PDLP()`.
The candidate is the basic solution when `x_B` lies within the bounds, and
the point after the pushes otherwise. It is written when its primal
violation is finite and at most `max(reltol, violation of the incoming
point)`. A non-finite value, or an exception, keeps the PDLP point.
"""
function crossover!(
        sol::PrimalDualSolution, milp::MILP, cx::SpiralCrossover;
        zero_tol::Real = 1.0e-8,
        reltol::Real = 1.0e-4,
        time_limit::Real = Inf,
    )
    if isbatched(milp)
        # One clock for the whole batch. A spent budget keeps the later PDLP points.
        deadline = current_time() + Float64(time_limit)
        for i in 1:nbinstances(milp)
            remain = deadline - current_time()
            remain > 0 || break
            crossover!(
                instance(sol, i), instance(milp, i), cx;
                zero_tol, reltol, time_limit = remain,
            )
        end
        return sol
    end
    n, m = nbvar(milp), nbcons(milp)
    m == 0 && return sol

    t0, limit = current_time(), Float64(time_limit)
    try
        viol0 = Float64(_scaled_violation(milp, sol.x))
        st = _push_once!(milp, sol, cx, zero_tol; t0, limit)
        x_push, y_push = copy(st.x), copy(st.y)
        basis_push, n_push = copy(st.basis), st.n_basis
        # A failed cleanup must not erase the pushes.
        function _restore_push!()
            copyto!(st.x, x_push)
            copyto!(st.y, y_push)
            copyto!(st.basis, basis_push)
            st.n_basis = n_push
            return nothing
        end

        # Skip the square cleanup when the wall budget is already gone
        # (it densifies A_B and has no per-call timeout).
        indep = false
        if !_time_up(st)
            indep = _linear_independence!(st)
        end
        if indep
            _improve_primal!(st)
            # A relative KKT pass can leave the absolute primal violation at
            # the PDLP level. The square solve is the step that removes it,
            # and it is refused when `x_B` is outside the bounds.
            _cleanup!(st) || _restore_push!()
            _snap_structural!(st, n)
        end
        if any(!isfinite, view(st.x, 1:n)) || any(!isfinite, st.y)
            @warn "SpiralCrossover produced non-finite values; keeping PDLP point"
            return sol
        end
        # `v` is the absolute primal violation. The candidate may be less
        # tight than the incoming point only up to `termination_reltol`.
        allow = max(Float64(reltol), viol0)
        viol1 = Float64(_scaled_violation(milp, view(st.x, 1:n)))
        if !isfinite(viol1) || viol1 > allow
            _restore_push!()
            _snap_structural!(st, n)
            viol1 = Float64(_scaled_violation(milp, view(st.x, 1:n)))
        end
        if !isfinite(viol1) || viol1 > allow
            @warn "SpiralCrossover primal violation above the incoming point; keeping PDLP point"
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
