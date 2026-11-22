"""
    GPUSparseMatrix

Union of the sparse formats that CoolPDLP implements with its own kernels.
"""
const GPUSparseMatrix = Union{GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL}

# The formats implement `sparse_mul!`, and `mul!` forwards to it with exactly the signatures of
# `sparse_mul!`: a broader one would be ambiguous with LinearAlgebra's generic `mul!`, and would
# also catch the products with mismatched storage that this generic `mul!` used to serve.
for (M, AV) in (
        (:GPUSparseMatrixCOO, :(GPUSparseMatrixCOO{T, Ti, V})),
        (:GPUSparseMatrixCSR, :(GPUSparseMatrixCSR{T, Ti, V})),
        (:GPUSparseMatrixELL, :(GPUSparseMatrixELL{T, Ti})),
    )
    @eval begin
        function LinearAlgebra.mul!(
                c::V, A::$AV, b::V, α::Number, β::Number
            ) where {T <: Number, Ti, V <: DenseVector{T}}
            return sparse_mul!(c, A, b, α, β)
        end
        function LinearAlgebra.mul!(
                c::DenseMatrix{T}, A::$M{T}, b::DenseMatrix{T}, α::Number, β::Number
            ) where {T <: Number}
            return sparse_mul!(c, A, b, α, β)
        end
    end
end

"""
    sametype_transpose(A::AbstractMatrix)

Return a matrix of the same type of `A` containing `transpose(A)` (as opposed to a `Transpose{...}` wrapper).

The default implementation is just `convert(typeof(A), transpose(A))` but it may need to be overloaded for certain matrix types.
"""
sametype_transpose(A::AbstractMatrix) = convert(typeof(A), transpose(A))

zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))
one!(x::AbstractArray) = fill!(x, one(eltype(x)))

# helpers for `mul!` implementations
struct One <: Number end
Base.:*(::One, x::Number) = x
struct Zero <: Number end
Base.:*(::Zero, ::Number) = Zero()
Base.:+(x::Number, ::Zero) = x

"""
    colnorm(x)

Return the Euclidean norm of `x`, or one norm per column if `x` is batched.
"""
colnorm(v::AbstractVector) = norm(v)
colnorm(m::AbstractMatrix) = norm.(eachcol(m))

"""
    colnorm!!(dest, x)

Compute the Euclidean norm of `x`, or one norm per column if `x` is batched, into `dest`.

Unlike [`colnorm`](@ref), the batched (`AbstractMatrix`) method computes `sqrt(sum(abs2, ...))`
directly instead of `LinearAlgebra.norm`'s scaled, overflow-safe algorithm: it needs to stay a
single allocation-free, GPU-broadcastable reduction, since it runs every iteration on the hot
path (see [`primal_weight_update!!`](@ref) and [`kkt_errors!`](@ref)). This means `colnorm!!` can
over/underflow on badly scaled columns where `colnorm` would not; that tradeoff is intentional
here, since the values it norms are primal-dual residuals that are not expected to approach the
extremes of the floating-point range in practice.
"""
colnorm!!(::Number, v::AbstractVector) = norm(v)
function colnorm!!(dest::AbstractVector, m::AbstractMatrix)
    sum!(abs2, transpose(dest), m)
    dest .= sqrt.(dest)
    return dest
end

"""
    coldot(a, b)

Return the scalar product of `a` and `b`, or one scalar product per column if either is batched.
"""
coldot(a::AbstractVector, b::AbstractVector) = dot(a, b)
coldot(a::AbstractVecOrMat, b::AbstractVecOrMat) = vec(sum(a .* b; dims = 1))

"""
    colsum!!(dest, x)

Compute the sum of `x`, or one sum per column if `x` is batched, into `dest`.
"""
colsum!!(::Number, v::AbstractVector) = sum(v)
function colsum!!(dest::AbstractVector, m::AbstractMatrix)
    sum!(transpose(dest), m)
    return dest
end

@inline positive_part(a::Number) = max(a, zero(a))
@inline negative_part(a::Number) = -min(a, zero(a))

@inline function safe(x::T) where {T <: AbstractFloat}
    if x == typemax(T)
        return prevfloat(x)
    elseif x == typemin(T)
        return nextfloat(x)
    else
        return x
    end
end

"""
    safeprod_left(left, right)

Compute `left * right`, except that `right` is forced to zero whenever `left` is infinite.

This is used to evaluate terms like `l * y⁺` in the dual objective, where `l` is a (possibly
infinite) constraint/variable bound and `y⁺` its associated Lagrange multiplier. The PDHG
update and [`proj_multiplier`](@ref) guarantee that the multiplier paired with an infinite
bound is mathematically zero, but not necessarily *exactly* zero in floating point (e.g. a
free row's multiplier is computed as a `σ`/`inv(σ)` round-trip that can leave a tiny nonzero
residual). Checking `iszero(right)` instead of always zeroing out would let such residuals
turn into `±Inf` (and, once several rows are summed, `Inf - Inf = NaN`) even though nothing
is actually wrong with the solution.
"""
@inline safeprod_left(left, right) = ifelse(isinf(left), zero(right), left * right)

"""
    proj_multiplier(λ, l, u)

Project `λ` onto the feasible space of the (double) Lagrange multiplier `λ⁺ - λ⁻` associated with the constraint `l ≤ x ≤ u`, where `l` and/or `u` might be infinite.
"""
@inline function proj_multiplier(λ::T, l::T, u::T) where {T <: Number}
    lmin = l == typemin(T)
    umax = u == typemax(T)
    return ifelse(
        lmin,
        ifelse(
            umax,
            zero(T),
            -negative_part(λ)
        ),
        ifelse(
            umax,
            positive_part(λ),
            λ
        )
    )
end

"""
    combine(l, u)

Return the largest finite absolute value between the two bounds, or zero if neither is finite.
"""
function combine(l::Number, u::Number)
    ls = ifelse(isfinite(l), abs(l), zero(l))
    us = ifelse(isfinite(u), abs(u), zero(u))
    return max(zero(l), ls, us)
end

"""
    Symmetrized

Represent a symmetric matrix `Kᵀ * K` lazily.
"""
struct Symmetrized{T <: Number, V <: DenseVector{T}, M <: AbstractMatrix{T}, Mt <: AbstractMatrix{T}}
    K::M
    Kᵀ::Mt
    scratch::V
end

function Symmetrized(K::AbstractMatrix, Kᵀ::AbstractMatrix)
    scratch = allocate(get_backend(K), eltype(K), size(K, 1))
    return Symmetrized(K, Kᵀ, scratch)
end

Base.eltype(sym::Symmetrized) = eltype(sym.K)
Base.size(sym::Symmetrized) = (n = size(sym.K, 2); (n, n))
Base.size(sym::Symmetrized, d::Int) = d in (1, 2) ? size(sym.K, 2) : 1

function LinearAlgebra.mul!(y, sym::Symmetrized, x)
    check_mul_dims(y, sym, x)
    (; K, Kᵀ, scratch) = sym
    mul!(scratch, K, x)
    mul!(y, Kᵀ, scratch)
    return y
end

"""
    abs_entries(A)

Return a matrix with the same sparsity structure (and type) as `A` whose stored entries are
the absolute values of those of `A`.
"""
abs_entries(A::AbstractMatrix) = abs.(A)
abs_entries(A::SparseMatrixCSC) = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, abs.(A.nzval))
abs_entries(A::GPUSparseMatrixCSR) = GPUSparseMatrixCSR(A.m, A.n, A.rowptr, A.colval, abs.(A.nzval))
abs_entries(A::GPUSparseMatrixCOO) = GPUSparseMatrixCOO(A.m, A.n, A.rowval, A.colval, abs.(A.nzval))
abs_entries(A::GPUSparseMatrixELL) = GPUSparseMatrixELL(A.m, A.n, A.colval, abs.(A.nzval))

"""
    spectral_norm_upper_bound(K, Kᵀ)

Return the cheap and rigorous upper bound `sqrt(‖K‖₁ ‖K‖∞) ≥ ‖K‖₂`, computed with two
matrix-vector products by `|K|` and `|Kᵀ|` so that it works on every backend.
"""
function spectral_norm_upper_bound(K::AbstractMatrix{T}, Kᵀ::AbstractMatrix{T}) where {T <: Number}
    m, n = size(K)
    (m == 0 || n == 0) && return zero(real(T))
    backend = get_backend(K)
    ones_n = allocate(backend, T, n)
    ones_m = allocate(backend, T, m)
    fill!(ones_n, one(T))
    fill!(ones_m, one(T))
    row_sums = allocate(backend, T, m)
    col_sums = allocate(backend, T, n)
    mul!(row_sums, abs_entries(K), ones_n)
    mul!(col_sums, abs_entries(Kᵀ), ones_m)
    norm_inf = real(maximum(row_sums))
    norm_1 = real(maximum(col_sums))
    return sqrt(norm_1 * norm_inf)
end

"""
    spectral_norm(K, Kᵀ; tol, maxiter, upper = false)

Estimate the spectral norm of `K` with the Lanczos method applied to `Kᵀ * K`, starting from
a deterministic random vector.

The iteration stops once the Lanczos residual bound `‖Kᵀ K y - θ y‖ ≤ tol * θ` holds for the
largest Ritz pair `(θ, y)`, or after `maxiter` products by `Kᵀ * K`. On convergence, it
returns `sqrt(θ)`, which lies below `‖K‖₂` (a Ritz value never exceeds the largest
eigenvalue) by at most about `tol / 2` relative when `θ` converged to the top of the spectrum;
with `upper = true`, it returns `sqrt(θ + residual)` instead, an upper bound of `‖K‖₂` in that
case, since an eigenvalue lies within `residual` of `θ`. `maxiter` should be
chosen independently of the size of `K` (see #95).

The largest Ritz value converges much faster than the power method's Rayleigh quotient when
the top of the spectrum is clustered (roughly like the power method on a matrix with the
square root of its condition number). This never throws on non-convergence: it then warns
(at every call, since each one is a different matrix that gets a conservative step size) and
returns the minimum between the inflated estimate `1.01 * sqrt(θ + residual)` and
the rigorous bound `sqrt(‖K‖₁ ‖K‖∞)`, so as to err on the large side (a smaller, safe step
size) rather than the small side (a divergent step size). Only the second bound is
guaranteed: the inflated estimate exceeds `‖K‖₂` whenever the top of the spectrum is
clustered (the usual reason for slow convergence), but not necessarily after very few
iterations.

This runs on the host with data-dependent control flow and is not meant to be traced by
Reactant: it is only called from `initialize`, outside of the traced `solve!` loop.

The element type of `K` must be `Float32` or `Float64` (possibly complex), and an
`ArgumentError` is thrown if `K` has a NaN or infinite entry.

Only local reorthogonalization is used: a loss of orthogonality creates spurious copies of
converged Ritz values, but not a wrong largest one, so full reorthogonalization (which costs
`O(maxiter * n)` memory) is not needed.
"""
function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        tol::Number,
        maxiter::Integer,
        upper::Bool = false,
    )
    T = eltype(K)
    R = real(T)
    # the Ritz values come from LAPACK's tridiagonal eigensolver
    R <: Union{Float32, Float64} ||
        throw(ArgumentError("spectral_norm needs Float32 or Float64 Lanczos vectors, got $T"))
    n = size(K, 2)
    (size(K, 1) == 0 || n == 0) && return zero(R)
    backend = get_backend(K)
    q = allocate(backend, T, n)
    q_cpu = adapt(CPU(), q)  # StableRNGs doesn't work on GPU
    randn!(StableRNG(0), q_cpu)
    copyto!(q, q_cpu)
    q ./= norm(q)
    q_prev = allocate(backend, T, n)
    fill!(q_prev, zero(T))
    w = allocate(backend, T, n)
    KᵀK = Symmetrized(K, Kᵀ)
    αs = R[]
    βs = R[]
    θ = zero(R)
    residual = typemax(R)
    converged = false
    β_prev = zero(R)
    for k in 1:maxiter
        mul!(w, KᵀK, q)
        α = real(dot(q, w))
        # a NaN would otherwise reach the step size silently through `min(NaN, bound)`
        isfinite(α) || throw(ArgumentError("spectral_norm: the matrix has NaN or infinite entries (or overflows in $T)"))
        w .-= α .* q .+ β_prev .* q_prev
        # one step of local reorthogonalization against `q` for numerical stability
        δ = real(dot(q, w))
        w .-= δ .* q
        α += δ
        β = R(norm(w))
        push!(αs, α)
        # an invariant subspace was found (e.g. `K == 0`): the Ritz values are exact
        breakdown = β <= eps(R) * max(abs(α), β_prev, eps(R))
        if breakdown || k == maxiter || k <= 10 || k % 5 == 0
            # largest Ritz pair of the tridiagonal Lanczos matrix, and the corresponding
            # residual bound `β * |last component of the Ritz vector|`
            F = eigen(SymTridiagonal(αs, βs), k:k)
            θ = max(zero(R), F.values[1])
            residual = β * abs(F.vectors[k, 1])
            if breakdown || residual <= R(tol) * θ
                converged = true
                break
            end
        end
        push!(βs, β)
        q_prev, q, w = q, w, q_prev
        q ./= β
        β_prev = β
    end
    if converged
        return upper ? sqrt(θ + residual) : sqrt(θ)
    else
        bound = R(spectral_norm_upper_bound(K, Kᵀ))
        estimate = min(R(1.01) * sqrt(θ + residual), bound)
        @warn "Spectral norm estimation did not converge within $maxiter iterations (relative residual $(residual / θ) > tol = $tol), using the conservative estimate $estimate (Lanczos value $(sqrt(θ)), rigorous bound $bound)"
        return estimate
    end
end

column_norm(A::AbstractMatrix, j::Integer, p) = norm(view(A, :, j), p)
column_norm(A::SparseMatrixCSC, j::Integer, p) = norm(view(nonzeros(A), nzrange(A, j)), p)

@inline _power_term(x, p) = iszero(x) ? zero(x) : abs(x)^p

"""
    column_power_sum(A, j, p)

Return `sum_i |A[i, j]|^p`. Zero entries are treated as contributing `0` (not `0^0 = 1`),
so the sparse and dense cases are aligned.
"""
column_power_sum(A::AbstractMatrix, j::Integer, p) =
    sum(x -> _power_term(x, p), view(A, :, j); init = zero(eltype(A)))
column_power_sum(A::SparseMatrixCSC, j::Integer, p) =
    sum(x -> _power_term(x, p), view(nonzeros(A), nzrange(A, j)); init = zero(eltype(A)))

mynnz(A::AbstractSparseMatrix) = nnz(A)
mynnz(A::AbstractMatrix) = prod(size(A))

indtype(::AbstractSparseMatrix{T, Ti}) where {T, Ti} = Ti

"""
    sparse_mul!(c, A, b, α, β)
    sparse_mul!(c::AbstractMatrix, A, b::AbstractMatrix, α, β; workgroupsize = nothing)

Kernel-launching body of `mul!` for the formats in [`GPUSparseMatrix`](@ref). Kept separate so
that the Reactant extension's overlay of `mul!` can reach it without calling `mul!` again.

For a batch of vectors, `workgroupsize` is passed on to the kernel launch, where `nothing` lets
KernelAbstractions choose it.
"""
function sparse_mul! end
