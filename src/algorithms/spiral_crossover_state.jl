# Liu–Lu Algo 2 lift: A_aug = [A  -I], A_aug [x; s] = 0 (full_row_rank).

const _BASIC = Int8(0)
const _AT_LB = Int8(1)
const _AT_UB = Int8(2)

"""Host working set for one spiral run. `T` is the algorithm eltype."""
mutable struct _SpiralState{T <: AbstractFloat}
    A::SparseMatrixCSC{T, Int}
    c::Vector{T}
    lv::Vector{T}
    uv::Vector{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}                # reduced costs c − Aᵀy
    basis::Vector{Int8}
    dual_active::Vector{Bool}
    n_basis::Int
    n_dual_active::Int
    n_basis_last::Int
    n_dual_active_last::Int
    b_work::Vector{T}
    c_work::Vector{T}
    zero_tol::T                 # = Algorithm `zero_tol`
    reltol::T                   # = Algorithm `termination_reltol`
    deadline::Float64           # absolute `time()`; `Inf` = no limit
end

_recount_basics!(st::_SpiralState) = (st.n_basis = count(==(_BASIC), st.basis); st)

"""True when `time_limit` elapsed (soft stop between pushes / stages)."""
_time_up(st::_SpiralState) = time() >= st.deadline

"""Map reduced cost / bounds → nonbasic side (`_BASIC` if free)."""
function _bound_side(zj, l, u)
    if isfinite(l) && isfinite(u)
        return zj >= 0 ? _AT_LB : _AT_UB
    elseif isfinite(l)
        return _AT_LB
    elseif isfinite(u)
        return _AT_UB
    else
        return _BASIC
    end
end

_finite_inf(::Type{T}, x::Real) where {T} = isfinite(x) ? T(x) : (x > 0 ? T(Inf) : T(-Inf))

# One host copy. A matching vector is copied, a matching CSC is reused, a device array is downloaded once.
_host_vec(::Type{T}, v::AbstractVector) where {T} = (cpu = v isa Vector{T} ? v : adapt(CPU(), v);
    cpu isa Vector{T} && cpu !== v ? cpu : Vector{T}(cpu))
_host_csc(::Type{T}, A::SparseMatrixCSC{T, Int}) where {T} = A
_host_csc(::Type{T}, A::AbstractMatrix) where {T} = (B = A isa SparseMatrixCSC ? A : SparseMatrixCSC(A);
    B isa SparseMatrixCSC{T, Int} ? B : SparseMatrixCSC{T, Int}(B))
_host_bounds(::Type{T}, v::AbstractVector) where {T} = _finite_inf.(T, v isa Vector ? v : adapt(CPU(), v))

"""Equality form `A_aug [x; s] = 0`, slack bounds `[lc, uc]`, in `eltype(x)`."""
function _lift_boxed(milp::MILP, x::AbstractVector{T}, y::AbstractVector) where {T <: AbstractFloat}
    A0 = _host_csc(T, milp.A)
    m, n = size(A0)
    xv = _host_vec(T, x)
    yv = _host_vec(T, y)
    length(xv) == n || throw(DimensionMismatch("x"))
    length(yv) == m || throw(DimensionMismatch("y"))

    A_aug = [A0 -spdiagm(0 => ones(T, m))]
    lv = [_host_bounds(T, milp.lv); _host_bounds(T, milp.lc)]
    uv = [_host_bounds(T, milp.uv); _host_bounds(T, milp.uc)]
    c = [_host_vec(T, milp.c); zeros(T, m)]
    s = clamp.(A0 * xv, view(lv, n + 1:n + m), view(uv, n + 1:n + m))
    x_aug = [xv; s]
    N = n + m
    return _SpiralState(
        A_aug, c, lv, uv, x_aug, yv, zeros(T, N),
        fill(_BASIC, N), fill(false, N),
        N, 0, N, 0, zeros(T, m), zeros(T, N), zero(T), zero(T),
        Inf,
    )
end

function _project_bounds!(st::_SpiralState)
    @. st.x = clamp(st.x, st.lv, st.uv)
    return st
end

function _update_reduced_costs!(st::_SpiralState)
    mul!(st.z, transpose(st.A), st.y)
    @. st.z = st.c - st.z
    return st
end

"""Fix nonbasics near a bound / reduced-cost soft test."""
function _fix_nonbasic!(st::_SpiralState, cx::SpiralCrossover)
    γ, ε = eltype(st.x)(cx.gamma), st.zero_tol
    st.n_basis_last = st.n_basis
    for j in eachindex(st.basis)
        st.basis[j] == _BASIC || continue
        l, u, xj, zj = st.lv[j], st.uv[j], st.x[j], st.z[j]
        if l == u
            st.basis[j] = _bound_side(zj, l, u)
            continue
        end
        at_lb = isfinite(l) && (xj <= l + ε || (xj - l) <= max(γ * zj, ε))
        at_ub = isfinite(u) && (xj >= u - ε || (u - xj) <= max(-γ * zj, ε))
        if at_lb ⊻ at_ub
            st.basis[j] = at_lb ? _AT_LB : _AT_UB
        elseif at_lb && at_ub
            st.basis[j] = _bound_side(zj, l, u)
        end
    end
    return _recount_basics!(st)
end

"""Snap nonbasics to finite bounds; set `b_work = −A_N x_N` (lift `b = 0`)."""
function _improve_primal!(st::_SpiralState)
    fill!(st.b_work, zero(eltype(st.b_work)))
    for j in eachindex(st.basis)
        if st.basis[j] == _AT_LB && isfinite(st.lv[j])
            st.x[j] = st.lv[j]
        elseif st.basis[j] == _AT_UB && isfinite(st.uv[j])
            st.x[j] = st.uv[j]
        elseif st.basis[j] != _BASIC
            st.basis[j] = _BASIC
        end
    end
    _recount_basics!(st)
    for j in eachindex(st.basis)
        st.basis[j] == _BASIC || axpy!(-st.x[j], view(st.A, :, j), st.b_work)
    end
    return st
end

function _perturb_cost!(st::_SpiralState{T}, cx::SpiralCrossover, rng) where {T}
    δ, nrm = T(cx.delta), norm(st.c, Inf) + one(T)
    for j in eachindex(st.c_work)
        r = T(rand(rng)) * δ
        isfinite(st.uv[j]) && !isfinite(st.lv[j]) && (r = -r)
        st.c_work[j] = r + st.c[j] / nrm
    end
    return st
end

function _perturb_rhs!(st::_SpiralState{T}, cx::SpiralCrossover, rng) where {T}
    δ = T(cx.delta)
    for i in eachindex(st.b_work)
        st.b_work[i] = T(rand(rng)) * δ
    end
    return st
end
