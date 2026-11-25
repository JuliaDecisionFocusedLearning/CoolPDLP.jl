"""
    Preconditioner

# Fields

$(TYPEDFIELDS)
"""
struct Preconditioner{T <: Number, V <: DenseVector{T}}
    "left preconditioner"
    D1::Diagonal{T, V}
    "right preconditioner"
    D2::Diagonal{T, V}
end

Preconditioner(milp::MILP) = Preconditioner(milp.D1, milp.D2)

function Base.:*(prec_out::Preconditioner, prec_in::Preconditioner)
    return Preconditioner(prec_out.D1 * prec_in.D1, prec_in.D2 * prec_out.D2)
end

Base.inv(prec::Preconditioner) = Preconditioner(inv(prec.D1), inv(prec.D2))

# Preconditioning effect

"""
    ConstraintMatrix

# Fields

$(TYPEDFIELDS)
"""
struct ConstraintMatrix{
        T <: Number, Ti <: Integer, M <: AbstractSparseMatrix{T, Ti}, MQ <: AbstractMatrix{T},
    }
    A::M
    At::M
    "quadratic objective matrix, scaled on both sides by the right preconditioner"
    Q::MQ
end

ConstraintMatrix(A::AbstractMatrix, At::AbstractMatrix) = ConstraintMatrix(A, At, ZeroMatrix{eltype(A)}(size(A, 2)))

function precondition(cons::ConstraintMatrix, prec::Preconditioner)
    (; A, At, Q) = cons
    (; D1, D2) = prec
    A_p = D1 * A * D2
    At_p = D2 * At * D1
    return ConstraintMatrix(A_p, At_p, scale_quadratic(Q, D2))
end

"""
    scale_quadratic(Q, D2)

Return `D2 * Q * D2`, the objective matrix in the variables `x = D2 * x_p`.
"""
scale_quadratic(Z::ZeroMatrix, ::Diagonal) = Z
scale_quadratic(Q::AbstractMatrix, D2::Diagonal) = D2 * Q * D2

function precondition(sol::PrimalDualSolution, prec::Preconditioner)
    (; x, y) = sol
    (; D1, D2) = prec
    x_p = D2 \ x
    y_p = D1 \ y
    return PrimalDualSolution(x_p, y_p)
end

function unprecondition(sol::PrimalDualSolution, prec::Preconditioner)
    x_p, y_p = sol.x, sol.y
    (; D1, D2) = prec
    x = D2 * x_p
    y = D1 * y_p
    return PrimalDualSolution(x, y)
end

function precondition(milp::MILP, prec::Preconditioner)
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    (; D1, D2) = prec
    cons = ConstraintMatrix(A, At, Q)
    cons_p = precondition(cons, prec)
    c_p = D2 * c
    lv_p, uv_p = D2 \ lv, D2 \ uv
    A_p, At_p = cons_p.A, cons_p.At
    lc_p, uc_p = D1 * lc, D1 * uc
    new_prec = prec * Preconditioner(milp)
    milp_p = MILP(;
        c = c_p,
        c0,  # the objective value is the same in both scales
        lv = lv_p,
        uv = uv_p,
        A = A_p,
        At = At_p,
        Q = cons_p.Q,
        lc = lc_p,
        uc = uc_p,
        D1 = new_prec.D1,
        D2 = new_prec.D2,
        bound_scale,
        obj_scale,
        int_var,
        var_names,
        con_names,
        dataset,
        name,
        path
    )
    return milp_p
end

# Preconditioner construction

"""
    stacked_column_norm(A, Q, j, p)

Return the `p`-norm of the `j`-th column of `[Q; A]`, as in PDQP and HPR-QP, which reduces to
that of `A` when `Q` is a [`ZeroMatrix`](@ref).
"""
stacked_column_norm(A::AbstractMatrix, ::ZeroMatrix, j::Integer, p) = column_norm(A, j, p)
function stacked_column_norm(A::AbstractMatrix, Q::AbstractMatrix, j::Integer, p)
    return norm((column_norm(A, j, p), column_norm(Q, j, p)), p)
end

"""
    stacked_column_power_sum(A, Q, j, p)

Return `∑ᵢ |Kᵢⱼ|ᵖ` over the `j`-th column of `K = [Q; A]`, which reduces to that of `A` when `Q`
is a [`ZeroMatrix`](@ref).
"""
stacked_column_power_sum(A::AbstractMatrix, ::ZeroMatrix, j::Integer, p) = column_power_sum(A, j, p)
function stacked_column_power_sum(A::AbstractMatrix, Q::AbstractMatrix, j::Integer, p)
    return column_power_sum(A, j, p) + column_power_sum(Q, j, p)
end

function identity_preconditioner(cons::ConstraintMatrix{T}) where {T}
    (; A) = cons
    d1 = ones(T, size(A, 1))
    d2 = ones(T, size(A, 2))
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function diagonal_norm_preconditioner(
        cons::ConstraintMatrix{T}; p_row::Number, p_col::Number
    ) where {T}
    (; A, At, Q) = cons
    # the columns of `Q` are scaled by the same factors as those of `A`
    col_norms = map(j -> stacked_column_norm(A, Q, j, p_col), axes(A, 2))
    row_norms = map(i -> column_norm(At, i, p_row), axes(A, 1))
    d1 = map(rn -> iszero(rn) ? one(T) : inv(sqrt(rn)), row_norms)
    d2 = map(cn -> iszero(cn) ? one(T) : inv(sqrt(cn)), col_norms)
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

"""
    chambolle_pock_preconditioner(cons; alpha)

Diagonal preconditioner from Lemma 2 of:

> Pock, Thomas, and Antonin Chambolle. "Diagonal preconditioning for first order
> primal-dual algorithms in convex optimization." 2011 International Conference
> on Computer Vision. IEEE, 2011. doi:10.1109/ICCV.2011.6126441

For matrix `K` of size m×n and `α ∈ [0, 2]`, the variable-side factor for
column `j` is `(sum_i |K[i,j]|^α)^{-1/2}` (over the column of `[Q; A]` for a quadratic program) and the constraint-side factor for
row `i` is `(sum_j |K[i,j]|^(2-α))^{-1/2}`. With these, `‖Σ^{1/2} K T^{1/2}‖ ≤ 1`
where `T, Σ` are the diagonals built from those factors.

Note that CoolPDLP uses `2-α` for the row and `α` for the column, to be aligned with
the 2021 paper "Practical large-scale linear programming using primal-dual hybrid gradient"
of Applegate et al. - this is the opposite of the convention used in
the original Pock-Chambolle paper referenced above.
"""
function chambolle_pock_preconditioner(cons::ConstraintMatrix{T}; alpha::Number) where {T}
    (; A, At, Q) = cons
    α_row = T(2 - alpha)
    α_col = T(alpha)
    # as in `diagonal_norm_preconditioner`, the columns of `Q` are scaled by the same factors as
    # those of `A`, so they count towards the column sums
    col_sums = map(j -> stacked_column_power_sum(A, Q, j, α_col), axes(A, 2))
    row_sums = map(i -> column_power_sum(At, i, α_row), axes(A, 1))
    d1 = map(s -> iszero(s) ? one(T) : inv(sqrt(s)), row_sums)
    d2 = map(s -> iszero(s) ? one(T) : inv(sqrt(s)), col_sums)
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

"""
    ruiz_preconditioner(cons; iterations)

Ruiz equilibration: for `iterations` rounds, rescale each row and column of `A`
(inside `cons`) by the square root of its inf norm.
"""
function ruiz_preconditioner(cons::ConstraintMatrix; iterations::Integer)
    prec = identity_preconditioner(cons)
    for _ in 1:iterations
        prec_next = diagonal_norm_preconditioner(cons; p_col = Inf, p_row = Inf)
        cons = precondition(cons, prec_next)
        prec = prec_next * prec
    end
    return prec
end

"""
    SCALING_FACTOR_BOUNDS

Range `(lo, hi)` to which the geometric-mean and Curtis-Reid factors are clamped, as in cuPDLPx.
"""
const SCALING_FACTOR_BOUNDS = (1.0e-20, 1.0e20)

"""
    geometric_mean_preconditioner(cons; iterations)

Geometric-mean scaling (Tomlin, 1975): alternately divide every row and then every column by
`sqrt(min * max)` of the absolute values of its nonzeros, `iterations` times.
"""
function geometric_mean_preconditioner(cons::ConstraintMatrix{T}; iterations::Integer) where {T}
    m, n = size(cons.A)
    prec = identity_preconditioner(cons)
    for _ in 1:iterations
        d1 = geometric_mean_factors(cons.At)
        prec_rows = Preconditioner(Diagonal(d1), Diagonal(ones(T, n)))
        cons = precondition(cons, prec_rows)
        d2 = geometric_mean_factors(cons.A)
        prec_cols = Preconditioner(Diagonal(ones(T, m)), Diagonal(d2))
        cons = precondition(cons, prec_cols)
        prec = prec_cols * prec_rows * prec
    end
    return prec
end

geometric_mean_factors(A::AbstractMatrix) = map(j -> geometric_mean_factor(A, j), axes(A, 2))

# `1 / sqrt(min * max)` over the finite nonzero magnitudes of column `j`, or one if there are none
function geometric_mean_factor(A::AbstractMatrix{T}, j::Integer) where {T}
    lo, hi = T(Inf), zero(T)
    for a in first(column_entries(A, j))
        v = abs(a)
        (v > 0 && isfinite(v)) || continue
        lo, hi = min(lo, v), max(hi, v)
    end
    iszero(hi) && return one(T)
    flo, fhi = SCALING_FACTOR_BOUNDS
    return clamp(inv(sqrt(lo) * sqrt(hi)), T(flo), T(fhi))
end

"""
    curtis_reid_preconditioner(cons; iterations)

Curtis-Reid least-squares scaling by `iterations` rounds of block coordinate descent (see the
math page of the documentation): the scaled matrix is `Diagonal(exp.(r)) * A * Diagonal(exp.(s))`.
"""
function curtis_reid_preconditioner(cons::ConstraintMatrix{T}; iterations::Integer) where {T}
    (; A, At) = cons
    m, n = size(A)
    r = zeros(T, m)
    s = zeros(T, n)
    for _ in 1:iterations
        for i in 1:m
            r[i] = curtis_reid_mean(At, i, s)
        end
        for j in 1:n
            s[j] = curtis_reid_mean(A, j, r)
        end
    end
    lo, hi = SCALING_FACTOR_BOUNDS
    d1 = clamp.(exp.(r), T(lo), T(hi))
    d2 = clamp.(exp.(s), T(lo), T(hi))
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

# mean of `-log|aᵢⱼ| - otherᵢ` over the finite nonzeros of column `j`, or zero if there are none.
# Stored zeros are skipped: their `-Inf` logarithm would send the factor to its clamp
function curtis_reid_mean(A::AbstractMatrix{T}, j::Integer, other::AbstractVector{T}) where {T}
    vals, idx = column_entries(A, j)
    acc, k = zero(T), 0
    for t in eachindex(vals, idx)
        v = abs(vals[t])
        (v > 0 && isfinite(v)) || continue
        acc += -log(v) - other[idx[t]]
        k += 1
    end
    return k == 0 ? zero(T) : acc / k
end

function column_entries(A::SparseMatrixCSC, j::Integer)
    rg = nzrange(A, j)
    return view(nonzeros(A), rg), view(rowvals(A), rg)
end
function column_entries(A::AbstractMatrix, j::Integer)
    idx = findall(!iszero, view(A, :, j))
    return view(A, idx, j), idx
end

"""
    PreconditioningParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PreconditioningParameters{T <: Number, I <: Number}
    "norm parameter in the Chambolle-pock preconditioner"
    chambolle_pock_alpha::T
    "iteration parameter in the Ruiz preconditioner"
    ruiz_iter::I
    "number of geometric-mean passes before Ruiz"
    geometric_mean_iter::I = 0
    "number of Curtis-Reid rounds before the geometric mean"
    curtis_reid_iter::I = 0
    "whether to rescale the bounds and the objective after preconditioning"
    bound_objective_rescaling::Bool = false
end

function Base.show(io::IO, params::PreconditioningParameters)
    (; chambolle_pock_alpha, ruiz_iter, geometric_mean_iter, curtis_reid_iter, bound_objective_rescaling) = params
    return print(io, "PreconditioningParameters: chambolle_pock_alpha=$chambolle_pock_alpha, ruiz_iter=$ruiz_iter, geometric_mean_iter=$geometric_mean_iter, curtis_reid_iter=$curtis_reid_iter, bound_objective_rescaling=$bound_objective_rescaling")
end

function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At, Q) = milp
    (; chambolle_pock_alpha, ruiz_iter, geometric_mean_iter, curtis_reid_iter) = params
    # Curtis-Reid and the geometric mean only look at `A`; `Q` follows the column scaling
    cons = ConstraintMatrix(A, At, Q)
    prec_cr = curtis_reid_preconditioner(cons; iterations = curtis_reid_iter)
    cons = precondition(cons, prec_cr)
    prec_gm = geometric_mean_preconditioner(cons; iterations = geometric_mean_iter)
    cons = precondition(cons, prec_gm)
    prec_r = ruiz_preconditioner(cons; iterations = ruiz_iter)
    cons_r = precondition(cons, prec_r)
    prec_cp = chambolle_pock_preconditioner(cons_r; alpha = chambolle_pock_alpha)
    prec = prec_cp * prec_r * prec_gm * prec_cr
    return prec
end

"""
    bound_objective_factors(milp)

Return the factors `(β, γ) = (1 / (1 + ‖b‖₂), 1 / (1 + ‖c‖₂))`, with `bᵢ = combine(lcᵢ, ucᵢ)` as
in the relative primal error, one per instance when the bounds or the objective are batched.
"""
function bound_objective_factors(milp::MILP{T}) where {T}
    (; c, lc, uc) = milp
    β = inv.(one(T) .+ colnorm(combine.(lc, uc)))
    γ = inv.(one(T) .+ colnorm(c))
    return β, γ
end

# multiply each column of `x` by its per-instance factor in `s` (a number or a vector)
scale_columns(x::AbstractVecOrMat, s::BatchedNumber) = x .* transpose(s)

"""
    rescale_bounds_objective(milp, β, γ)

Return `milp` with bounds multiplied by `β` and objective by `γ`, factors recorded in
`milp.bound_scale` and `milp.obj_scale`. A solution `(x, y)` of `milp` becomes `(β x, γ y)`.
For that, a quadratic objective matrix becomes `(γ / β) Q`, see [`rescale_quadratic`](@ref).
"""
function rescale_bounds_objective(milp::MILP, β::BatchedNumber, γ::BatchedNumber)
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = scale_columns(c, γ),
        c0,
        lv = scale_columns(lv, β),
        uv = scale_columns(uv, β),
        A,
        At,
        Q = rescale_quadratic(Q, β, γ),
        lc = scale_columns(lc, β),
        uc = scale_columns(uc, β),
        D1,
        D2,
        bound_scale = bound_scale .* β,
        obj_scale = obj_scale .* γ,
        int_var,
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end

"""
    rescale_quadratic(Q, β, γ)

Return the objective matrix `(γ / β) Q` of the rescaled problem, whose objective is `β γ` times
the original one in the variables `β x`.

`Q` is shared by the whole batch, so with per-instance factors the result is a
[`BatchScaledMatrix`](@ref) holding one ratio per instance. That choice depends on the types of
`β` and `γ` only, not on whether the ratios happen to be equal, so that it stays type-stable.
"""
rescale_quadratic(Z::ZeroMatrix, ::BatchedNumber, ::BatchedNumber) = Z
rescale_quadratic(Z::ZeroMatrix, ::Number, ::Number) = Z
function rescale_quadratic(Q::AbstractMatrix, β::Number, γ::Number)
    r = γ / β
    return isone(r) ? Q : r * Q
end
function rescale_quadratic(Q::AbstractMatrix, β::BatchedNumber, γ::BatchedNumber)
    # in the element type of `Q`, whatever that of the factors
    ratio = collect(eltype(Q), broadcast(/, γ, β))
    return BatchScaledMatrix(Q, ratio)
end
# rescaling an already batch-scaled matrix (e.g. a second pass) multiplies the ratios
function rescale_quadratic(S::BatchScaledMatrix, β::BatchedNumber, γ::BatchedNumber)
    return BatchScaledMatrix(S.Q, convert.(eltype(S.Q), S.scale .* broadcast(/, γ, β)))
end
rescale_quadratic(S::BatchScaledMatrix, β::Number, γ::Number) = BatchScaledMatrix(S.Q, convert.(eltype(S.Q), S.scale .* (γ / β)))

function rescale_bounds_objective(sol::PrimalDualSolution, β::BatchedNumber, γ::BatchedNumber)
    return PrimalDualSolution(scale_columns(sol.x, β), scale_columns(sol.y, γ))
end
