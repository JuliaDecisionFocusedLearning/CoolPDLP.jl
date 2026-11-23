"""
    ZeroMatrix{T}(n)

Square `n × n` matrix of zeros, the default quadratic objective `Q` of a [`MILP`](@ref).

Every quadratic computation dispatches on it and reduces to the linear one at compile time, so
a linear program pays nothing for the possibility of a quadratic objective.
"""
struct ZeroMatrix{T <: Number} <: AbstractMatrix{T}
    n::Int
end

Base.size(Z::ZeroMatrix) = (Z.n, Z.n)
Base.getindex(::ZeroMatrix{T}, ::Int, ::Int) where {T} = zero(T)
Base.isapprox(Z1::ZeroMatrix, Z2::ZeroMatrix; kwargs...) = size(Z1) == size(Z2)
Adapt.adapt_structure(_, Z::ZeroMatrix) = Z
SparseArrays.nnz(::ZeroMatrix) = 0
mynnz(::ZeroMatrix) = 0

"""
    BatchScaledMatrix(Q, scale)

The objective matrix `scaleᵢ * Q` of instance `i` of a batch, with `Q` shared and one positive
factor per instance in the vector `scale`.

It comes from the bound and objective rescaling of a batched quadratic program whose instances
need different ratios `γ / β` (see [`rescale_quadratic`](@ref)). Every product by the objective
matrix goes through [`quadratic_mul!`](@ref), which scales the columns of the result, so the
algorithms need no change. Spectral norms take the largest factor (see [`quadratic_norm`](@ref)),
which keeps the step sizes safe for every instance.
"""
struct BatchScaledMatrix{T <: Number, M <: AbstractMatrix{T}, V <: AbstractVector} <: AbstractMatrix{T}
    "shared matrix"
    Q::M
    "positive factor of each instance"
    scale::V
end

Base.size(S::BatchScaledMatrix) = size(S.Q)
Base.getindex(::BatchScaledMatrix, ::Int, ::Int) = throw(ArgumentError("a BatchScaledMatrix has one value per instance"))
Base.iszero(S::BatchScaledMatrix) = iszero(S.Q)
Base.isapprox(S1::BatchScaledMatrix, S2::BatchScaledMatrix; kwargs...) = isapprox(S1.Q, S2.Q; kwargs...) && isapprox(S1.scale, S2.scale; kwargs...)
Adapt.adapt_structure(to, S::BatchScaledMatrix) = BatchScaledMatrix(adapt(to, S.Q), adapt(to, S.scale))
SparseArrays.nnz(S::BatchScaledMatrix) = nnz(S.Q)
mynnz(S::BatchScaledMatrix) = mynnz(S.Q)

"""
    quadratic_mul!(dest, Q, x)

Return `Q * x` in `dest`, scaling column `i` by `Q.scale[i]` for a [`BatchScaledMatrix`](@ref).
"""
quadratic_mul!(dest::AbstractVecOrMat, Q::AbstractMatrix, x::AbstractVecOrMat) = mul!(dest, Q, x)
function quadratic_mul!(dest::AbstractMatrix, S::BatchScaledMatrix, x::AbstractMatrix)
    mul!(dest, S.Q, x)
    dest .*= transpose(S.scale)
    return dest
end

"""
    instance_quadratic(Q, i)

Return the objective matrix of instance `i`: `Q` itself unless it is a [`BatchScaledMatrix`](@ref).
"""
instance_quadratic(Q::AbstractMatrix, ::Int) = Q
instance_quadratic(S::BatchScaledMatrix, i::Int) = Array(S.scale)[i] * S.Q

"""
    isquadratic(Q)

Return whether the objective matrix `Q` may be nonzero.

This only depends on the type of `Q` for a [`ZeroMatrix`](@ref) (never) and a dense matrix
(always). A sparse `Q` without any stored entry, which is what a linear program read from a
file gets (see [`MILP`](@ref)), is checked at run time: every quadratic computation below then
skips its products by `Q` and returns zeros of the same type as for a nonzero `Q`, so the code
stays type-stable while a linear program pays nothing but this check. That check only reads
host-side structure (the storage size, never the stored indices), so it stays a plain `Bool`
when the arrays of `Q` are traced by Reactant.
"""
isquadratic(::ZeroMatrix) = false
isquadratic(::AbstractMatrix) = true
isquadratic(Q::AbstractSparseMatrix) = nnz(Q) > 0
# `nnz` counts the stored column indices, a traced value under Reactant; the padded width is the
# length of the longest row, which is zero exactly when there is no stored entry
isquadratic(Q::GPUSparseMatrixELL) = !isempty(Q.colval)
isquadratic(S::BatchScaledMatrix) = isquadratic(S.Q)

"""
    backend_args(Q)

Return the arrays of the objective matrix `Q` whose backend must match the rest of the problem.
"""
backend_args(::ZeroMatrix) = ()
backend_args(Q::AbstractMatrix) = (Q,)
backend_args(S::BatchScaledMatrix) = (S.Q, S.scale)

"""
    NoQuadraticTerm

Value of `xᵀQx` when `Q` is a [`ZeroMatrix`](@ref): adding it to anything is a no-op.
"""
struct NoQuadraticTerm end

"""
    add_half!!(dest, q)
    subtract_half!!(dest, q)

Compute `dest ± q / 2`, column by column, into `dest`. A no-op when `q` is a [`NoQuadraticTerm`](@ref).
"""
add_half!!(dest::BatchedNumber, ::NoQuadraticTerm) = dest
add_half!!(dest::BatchedNumber, q::BatchedNumber) = broadcast!!((d, q) -> d + q / 2, dest, dest, q)
subtract_half!!(dest::BatchedNumber, ::NoQuadraticTerm) = dest
subtract_half!!(dest::BatchedNumber, q::BatchedNumber) = broadcast!!((d, q) -> d - q / 2, dest, dest, q)

"""
    scaled_quadratic_term(xQx, β, γ)

Return `xᵀQx / (β γ)`, the quadratic term of the original problem from that of the problem with
bounds rescaled by `β` and objective by `γ`; a [`NoQuadraticTerm`](@ref) stays one.
"""
scaled_quadratic_term(q::NoQuadraticTerm, ::BatchedNumber, ::BatchedNumber) = q
function scaled_quadratic_term(q::BatchedNumber, β::BatchedNumber, γ::BatchedNumber)
    return batched_apply!!((q, b, g) -> q / (b * g), q, q, β, γ)
end

"""
    quadratic_form!!(dest, scratch, Q, x)

Return `xᵀQx`, column by column, into `dest`, using the vector `scratch` for `Q * x`. Return a
[`NoQuadraticTerm`](@ref) without touching anything when `Q` is a [`ZeroMatrix`](@ref).
"""
function quadratic_form!!(::BatchedNumber, ::AbstractVecOrMat, ::ZeroMatrix, ::AbstractVecOrMat)
    return NoQuadraticTerm()
end
function quadratic_form!!(
        dest::BatchedNumber, scratch::AbstractVecOrMat, Q::AbstractMatrix, x::AbstractVecOrMat
    )
    isquadratic(Q) || return broadcast!!(zero, dest, dest)
    Qx = quadratic_mul!(scratch, Q, x)
    return colsum!!(dest, @. scratch = x * Qx)
end

"""
    add_quadratic_gradient!!(dest, scratch, g, Q, x)

Add `Q * x` to `g` in place and return `xᵀQx`, column by column, into `dest`, using the vector
`scratch` for `Q * x`. Return a [`NoQuadraticTerm`](@ref) without touching anything when `Q` is
a [`ZeroMatrix`](@ref).
"""
function add_quadratic_gradient!!(
        ::BatchedNumber, ::AbstractVecOrMat, ::AbstractVecOrMat, ::ZeroMatrix, ::AbstractVecOrMat
    )
    return NoQuadraticTerm()
end
function add_quadratic_gradient!!(
        dest::BatchedNumber, scratch::AbstractVecOrMat, g::AbstractVecOrMat,
        Q::AbstractMatrix, x::AbstractVecOrMat,
    )
    isquadratic(Q) || return broadcast!!(zero, dest, dest)
    Qx = quadratic_mul!(scratch, Q, x)
    g .+= Qx
    return colsum!!(dest, @. scratch = x * Qx)
end

"""
    mul_quadratic!(dest, Q, x)

Return `Q * x` in `dest`, which a [`ZeroMatrix`](@ref) or an empty sparse `Q` just zeroes.
"""
mul_quadratic!(dest::AbstractVecOrMat, ::ZeroMatrix, ::AbstractVecOrMat) = zero!(dest)
function mul_quadratic!(dest::AbstractVecOrMat, Q::AbstractMatrix, x::AbstractVecOrMat)
    isquadratic(Q) || return zero!(dest)
    return quadratic_mul!(dest, Q, x)
end

"""
    subtract_quadratic_gradient!(g, Q, x)

Compute `g - Q * x` in place, or leave `g` untouched when `Q` is a [`ZeroMatrix`](@ref).
"""
subtract_quadratic_gradient!(g::AbstractVecOrMat, ::ZeroMatrix, ::AbstractVecOrMat) = g
function subtract_quadratic_gradient!(g::AbstractVecOrMat, Q::AbstractMatrix, x::AbstractVecOrMat)
    isquadratic(Q) || return g
    return mul!(g, Q, x, -1, true)
end
function subtract_quadratic_gradient!(g::AbstractMatrix, S::BatchScaledMatrix, x::AbstractMatrix)
    isquadratic(S) || return g
    # g - s Q x = s (g / s - Q x), in place and without scratch (the factors are positive)
    g ./= transpose(S.scale)
    mul!(g, S.Q, x, -1, true)
    g .*= transpose(S.scale)
    return g
end

"""
    quadratic_value(x, Q)

Return `½ xᵀQx`, one value per column if `x` is batched, or `false` for a [`ZeroMatrix`](@ref).

It allocates `Q * x`: it is meant for reporting objective values, not for the solve loop.
"""
quadratic_value(::AbstractVecOrMat, ::ZeroMatrix) = false
quadratic_value(x::AbstractVecOrMat, Q::AbstractMatrix) = coldot(x, Q * x) ./ 2
quadratic_value(x::AbstractMatrix, S::BatchScaledMatrix) = coldot(x, S.Q * x) .* S.scale ./ 2

"""
    check_convex(Q)

Throw an `ArgumentError` if the symmetric matrix `Q` is detectably not positive semidefinite:
a negative diagonal entry, or an off-diagonal entry with `Qᵢⱼ² > Qᵢᵢ Qⱼⱼ` (a 2×2 principal minor
of the wrong sign). These are necessary conditions only, cheap to check on the nonzeros.

Only a sparse `Q` is checked, and only by the QPS and MOI paths: a dense `Q`, or any `Q` passed
to the keyword constructor of [`MILP`](@ref), is trusted to be convex.
"""
check_convex(::ZeroMatrix) = nothing
function check_convex(Q::SparseMatrixCSC)
    d = diag(Q)
    rows, cols, vals = findnz(Q)
    for (i, j, v) in zip(rows, cols, vals)
        if i == j
            v < 0 && throw(ArgumentError("Nonconvex quadratic objective: Q[$i, $i] = $v < 0"))
        elseif v^2 > d[i] * d[j] * (1 + sqrt(eps(float(typeof(v)))))
            throw(
                ArgumentError(
                    "Nonconvex quadratic objective: Q[$i, $j]² > Q[$i, $i] * Q[$j, $j]"
                )
            )
        end
    end
    return nothing
end
