# Primal / dual OLS pushes (Algo 2).

_soft_zero(v, ε) = abs(v) <= ε ? zero(v) : v

"""Ratio test with orientation flip; `Inf` if neither direction hits a bound."""
function _oriented_theta(v::AbstractVector, ratio, ε::Real)
    θ = ratio(v, ε)
    if !isfinite(θ)
        v .*= -1
        θ = ratio(v, ε)
    end
    return θ
end

"""Least squares on columns `cols` of the host lift, by [`ols!`](@ref)."""
function _spiral_ols_cols!(
        xB::AbstractVector,
        st::_SpiralState,
        cols::AbstractVector{Int},
        b::AbstractVector,
    )
    isempty(cols) && return xB
    return ols!(xB, st.A[:, cols], b)
end

"""Write soft-thresholded `scale * residual[k]` into `dest[cols[k]]`."""
function _soft_fill!(dest::AbstractVector, cols, residual, scale::Real, ε::Real)
    fill!(dest, zero(eltype(dest)))
    for (k, j) in enumerate(cols)
        dest[j] = _soft_zero(scale * residual[k], ε)
    end
    return dest
end

"""Primal ray: OLS `min ‖A_B u − A_B c̃_B‖`, then `v_B = −(η/ω)(c̃_B − u)`."""
function _primal_ray!(v::AbstractVector, st::_SpiralState, cx::SpiralCrossover, basic::BitVector)
    B = findall(basic)
    isempty(B) && return v
    T = eltype(st.x)
    cB = st.c_work[B]
    rhs = zeros(T, size(st.A, 1))
    mul!(rhs, st.A[:, B], cB)
    ATy = zeros(T, length(B))
    _spiral_ols_cols!(ATy, st, B, rhs)
    ηω = T(cx.eta) / T(cx.omega)
    return _soft_fill!(v, B, cB .- ATy, -ηω, st.zero_tol)
end

function _primal_ratio(st::_SpiralState, v::AbstractVector, basic::BitVector, ε::Real)
    θ = eltype(v)(Inf)
    for j in eachindex(v)
        basic[j] || continue
        vj = v[j]
        if vj < -ε
            θ = min(θ, (st.lv[j] - st.x[j]) / vj)
        elseif vj > ε
            θ = min(θ, (st.uv[j] - st.x[j]) / vj)
        end
    end
    return θ
end

function _move_x!(st::_SpiralState, cx::SpiralCrossover)
    basic = st.basis .== _BASIC
    v = zeros(eltype(st.x), length(st.x))
    _primal_ray!(v, st, cx, basic)
    ε = st.zero_tol
    norm(v) <= ε && return
    θ = _oriented_theta(v, (vv, ee) -> _primal_ratio(st, vv, basic, ee), ε)
    if isfinite(θ)
        @. st.x = st.x + θ * v
        _project_bounds!(st)
    end
    return
end

function _primal_push!(st::_SpiralState, cx::SpiralCrossover, rng)
    _update_reduced_costs!(st)
    _fix_nonbasic!(st, cx)
    _improve_primal!(st)
    pushes = 0
    while pushes <= 1 || st.n_basis < st.n_basis_last
        pushes >= cx.max_pushes && break
        _time_up(st) && break
        st.n_basis == 0 && break
        _perturb_cost!(st, cx, rng)
        _move_x!(st, cx)
        pushes += 1
        _update_reduced_costs!(st)
        _fix_nonbasic!(st, cx)
        _improve_primal!(st)
        # stop when basic count no longer decreases
        if st.n_basis <= size(st.A, 1) && st.n_basis >= st.n_basis_last
            break
        end
    end
    return st
end

"""
Mark dual-active columns (near-zero reduced cost on free / one-sided vars).

The bound side of a nonbasic column stays the one chosen from where `x` sits.
Retagging it from the sign of the reduced cost would flip [`_dual_ratio`](@ref).
"""
function _cal_dual_active!(st::_SpiralState)
    ε = st.zero_tol
    st.n_dual_active_last = st.n_dual_active
    _update_reduced_costs!(st)
    for j in eachindex(st.dual_active)
        s = st.z[j]
        has_l, has_u = isfinite(st.lv[j]), isfinite(st.uv[j])
        active = st.basis[j] == _BASIC || st.dual_active[j]
        if has_l && !has_u
            active |= s <= ε
        elseif has_u && !has_l
            active |= s >= -ε
        else
            active |= abs(s) <= ε
        end
        st.dual_active[j] = active
    end
    st.n_dual_active = count(st.dual_active)
    return st
end

"""Dual spiral ray on the dual-active set D: `v_y = η ω (b̃ − A_D (A_Dᵀ A_D)† A_Dᵀ b̃)`."""
function _dual_ray!(vy::AbstractVector, st::_SpiralState, cx::SpiralCrossover)
    D = findall(st.dual_active)
    isempty(D) && return vy
    T = eltype(vy)
    xv = zeros(T, length(D))
    _spiral_ols_cols!(xv, st, D, st.b_work)
    AxD = zeros(T, length(vy))
    mul!(AxD, st.A[:, D], xv)
    ηω = T(cx.eta) * T(cx.omega)
    return _soft_fill!(vy, eachindex(vy), st.b_work .- AxD, ηω, st.zero_tol)
end

function _dual_ratio(st::_SpiralState, vy::AbstractVector, ε::Real)
    atv = transpose(st.A) * vy
    θ = eltype(vy)(Inf)
    for j in eachindex(st.dual_active)
        st.dual_active[j] && continue
        s, a = st.z[j], atv[j]
        if st.basis[j] == _AT_UB && a < -ε
            θ = min(θ, s / a)
        elseif st.basis[j] == _AT_LB && a > ε
            θ = min(θ, s / a)
        end
    end
    return θ
end

function _move_y!(st::_SpiralState, cx::SpiralCrossover)
    vy = zeros(eltype(st.y), length(st.y))
    _dual_ray!(vy, st, cx)
    ε = st.zero_tol
    norm(vy) <= ε && return
    θ = _oriented_theta(vy, (vv, ee) -> _dual_ratio(st, vv, ee), ε)
    isfinite(θ) && (@. st.y = st.y + θ * vy)
    return
end

function _dual_push!(st::_SpiralState, cx::SpiralCrossover, rng)
    _cal_dual_active!(st)
    pushes = 0
    while pushes <= 1 || st.n_dual_active > st.n_dual_active_last
        pushes >= cx.max_pushes && break
        _time_up(st) && break
        _perturb_rhs!(st, cx, rng)
        _move_y!(st, cx)
        pushes += 1
        _cal_dual_active!(st)
        st.n_dual_active <= st.n_dual_active_last && break
    end
    return st
end
