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
    colnorm!!(dest, x, norm)

Compute the norm of `x` selected by the symbol `norm` (`:L2` or `:Linf`), or one norm per column
if `x` is batched, into `dest`.

The choice is a plain symbol rather than a number so that Reactant does not trace it: the branch
is resolved once, when tracing, and only the selected reduction ends up in the compiled program.
Any other symbol throws an `ArgumentError`, rather than silently falling back to `:L2` (a
`TerminationParameters` built directly skips [`check_termination_parameters`](@ref)).
"""
function colnorm!!(dest, x::AbstractVecOrMat, norm::Symbol)
    if norm === :Linf
        return colmaxabs!!(dest, x)
    elseif norm === :L2
        return colnorm!!(dest, x)
    else
        throw(ArgumentError("unknown norm $(repr(norm)), expected :L2 or :Linf"))
    end
end

"""
    colmaxabs!!(dest, x)

Compute the largest absolute value (the L∞ norm) of `x`, or one per column if `x` is batched,
into `dest`. An empty column has norm zero.
"""
colmaxabs!!(::Number, v::AbstractVector{T}) where {T} = mapreduce(abs, max, v; init = zero(T))
function colmaxabs!!(dest::AbstractVector, m::AbstractMatrix)
    # the absolute values are nonnegative, so zero is a neutral starting point for `max`
    fill!(dest, zero(eltype(dest)))
    Base.mapreducedim!(abs, max, transpose(dest), m)
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
    spectral_norm(K, Kᵀ; tol, maxiter)

Compute the spectral norm of `K` with the power method from IterativeSolvers.jl, using
`tol` as the absolute tolerance on the estimated eigenvalue of `Kᵀ * K` and `maxiter` as
the iteration budget.

Throws an `AssertionError` if the power method fails to converge within `maxiter`
iterations, instead of silently returning an inaccurate estimate. `maxiter` should be
chosen independently of the size of `K`: IterativeSolvers' own default (`size(B, 2)`)
ties the budget to problem size instead of the power method's convergence rate, which is
exactly what let the estimate diverge silently on large matrices (#95).
"""
function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        tol::Number,
        maxiter::Integer,
    )
    x0 = allocate(get_backend(K), eltype(K), size(K, 2))
    x0_cpu = adapt(CPU(), x0)  # StableRNGs doesn't work on GPU
    randn!(StableRNG(0), x0_cpu)
    copyto!(x0, x0_cpu)
    # normalize initial guess following the docstring of `powm!`
    x0 ./= norm(x0)
    KᵀK = Symmetrized(K, Kᵀ)
    λ, _, powm_history = powm!(KᵀK, x0; tol, maxiter, log = true)
    # `λ` is a Rayleigh quotient of the PSD matrix `KᵀK`, so it is mathematically
    # nonnegative; `>= zero(λ)` (not `>`) so a legitimately zero spectral norm (no
    # constraint rows, or an all-zero `K`) is not mistaken for the floating-point-noise
    # case this guards against, which would otherwise reach `sqrt` as a small negative
    @assert λ >= zero(λ)
    @assert powm_history.isconverged
    return sqrt(λ)
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
