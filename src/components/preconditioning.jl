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
struct ConstraintMatrix{T <: Number, Ti <: Integer, M <: AbstractSparseMatrix{T, Ti}}
    A::M
    At::M
end

function precondition(cons::ConstraintMatrix, prec::Preconditioner)
    (; A, At) = cons
    (; D1, D2) = prec
    A_p = D1 * A * D2
    At_p = D2 * At * D1
    return ConstraintMatrix(A_p, At_p)
end

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
        c, c0, lv, uv, A, At, lc, uc, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    (; D1, D2) = prec
    cons = ConstraintMatrix(A, At)
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

function identity_preconditioner(cons::ConstraintMatrix{T}) where {T}
    (; A) = cons
    d1 = ones(T, size(A, 1))
    d2 = ones(T, size(A, 2))
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function diagonal_norm_preconditioner(
        cons::ConstraintMatrix{T}; p_row::Number, p_col::Number
    ) where {T}
    (; A, At) = cons
    col_norms = map(j -> column_norm(A, j, p_col), axes(A, 2))
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
column `j` is `(sum_i |K[i,j]|^α)^{-1/2}` and the constraint-side factor for
row `i` is `(sum_j |K[i,j]|^(2-α))^{-1/2}`. With these, `‖Σ^{1/2} K T^{1/2}‖ ≤ 1`
where `T, Σ` are the diagonals built from those factors.

Note that CoolPDLP uses `2-α` for the row and `α` for the column, to be aligned with
the 2021 paper "Practical large-scale linear programming using primal-dual hybrid gradient"
of Applegate et al. - this is the opposite of the convention used in
the original Pock-Chambolle paper referenced above.
"""
function chambolle_pock_preconditioner(cons::ConstraintMatrix{T}; alpha::Number) where {T}
    (; A, At) = cons
    α_row = T(2 - alpha)
    α_col = T(alpha)
    col_sums = map(j -> column_power_sum(A, j, α_col), axes(A, 2))
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

Range `(lo, hi)` to which the factors of each geometric-mean and Curtis-Reid pass are clamped,
that of cuPDLPx. Both ends stay finite and nonzero in `Float32`.
"""
const SCALING_FACTOR_BOUNDS = (1.0e-20, 1.0e20)

"""
    geometric_mean_preconditioner(cons; iterations)

Geometric-mean scaling (Tomlin, 1975), as in cuPDLPx: alternately divide every row and then
every column by `sqrt(min * max)` of the absolute values of its nonzeros, `iterations` times.
"""
function geometric_mean_preconditioner(cons::ConstraintMatrix; iterations::Integer)
    prec = identity_preconditioner(cons)
    for _ in 1:iterations
        prec_rows = geometric_mean_step(cons, Val(:rows))
        cons = precondition(cons, prec_rows)
        prec_cols = geometric_mean_step(cons, Val(:cols))
        cons = precondition(cons, prec_cols)
        prec = prec_cols * prec_rows * prec
    end
    return prec
end

function geometric_mean_step(cons::ConstraintMatrix{T}, ::Val{side}) where {T, side}
    (; A, At) = cons
    if side == :rows
        d1 = map(i -> geometric_mean_factor(At, i), axes(A, 1))
        d2 = ones(T, size(A, 2))
    else
        d1 = ones(T, size(A, 1))
        d2 = map(j -> geometric_mean_factor(A, j), axes(A, 2))
    end
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

"""
    geometric_mean_factor(A, j)

Return `1 / sqrt(min * max)` over the finite nonzero absolute values of column `j` of `A`,
clamped to [`SCALING_FACTOR_BOUNDS`](@ref) like cuPDLPx, or one if the column is empty.
"""
function geometric_mean_factor(A::AbstractMatrix{T}, j::Integer) where {T}
    lo, hi = T(Inf), zero(T)
    for a in column_values(A, j)
        v = abs(a)
        (v > 0 && isfinite(v)) || continue
        lo, hi = min(lo, v), max(hi, v)
    end
    iszero(hi) && return one(T)
    flo, fhi = SCALING_FACTOR_BOUNDS
    return clamp(inv(sqrt(lo) * sqrt(hi)), T(flo), T(fhi))
end

column_values(A::AbstractMatrix, j::Integer) = view(A, :, j)
column_values(A::SparseMatrixCSC, j::Integer) = view(nonzeros(A), nzrange(A, j))

"""
    curtis_reid_preconditioner(cons; iterations)

Least-squares scaling with the objective of Curtis and Reid (1972): approximately minimize
`∑ᵢⱼ (log|aᵢⱼ| + rᵢ + sⱼ)²` over the nonzeros, by `iterations` rounds of block coordinate descent
(exact minimization in all of `r`, then in all of `s`). This is not the conjugate-gradient
method of Curtis and Reid, which converges faster: the descent decreases the objective at every
round but may stop well short of its minimum on badly scaled matrices. The scaled matrix is
`Diagonal(exp.(r)) * A * Diagonal(exp.(s))`, with factors clamped to [`SCALING_FACTOR_BOUNDS`](@ref).
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

"""
    curtis_reid_mean(A, j, other)

Return the mean of `-log|aᵢⱼ| - otherᵢ` over the nonzeros of column `j` of `A`, or zero if the
column has none. Explicitly stored zeros are skipped, as in [`geometric_mean_factor`](@ref):
their logarithm is `-Inf`, which would send the scaling factor of the column to its clamp.
"""
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
    "number of geometric-mean passes, applied before Ruiz (0 to disable, cuPDLPx uses 12)"
    geometric_mean_iter::I = 0
    "number of rounds of Curtis-Reid least-squares scaling, by block coordinate descent, applied before Ruiz (0 to disable, HPR-LP uses 20)"
    curtis_reid_iter::I = 0
    "whether to rescale bounds by `1 / (1 + ‖b‖)` and the objective by `1 / (1 + ‖c‖)` at the end"
    bound_objective_rescaling::Bool = false
    "whether `‖b‖` in bound rescaling sums `l² + u²` over the finite bounds of each row (cuPDLPx) instead of taking the largest finite `|l|`, `|u|` per row (HPR-LP)"
    bound_norm_sum_squares::Bool = false
end

function Base.show(io::IO, params::PreconditioningParameters)
    (;
        chambolle_pock_alpha, ruiz_iter, geometric_mean_iter, curtis_reid_iter,
        bound_objective_rescaling, bound_norm_sum_squares,
    ) = params
    return print(io, "PreconditioningParameters: chambolle_pock_alpha=$chambolle_pock_alpha, ruiz_iter=$ruiz_iter, geometric_mean_iter=$geometric_mean_iter, curtis_reid_iter=$curtis_reid_iter, bound_objective_rescaling=$bound_objective_rescaling, bound_norm_sum_squares=$bound_norm_sum_squares")
end

"""
    pdlp_preconditioner(milp, params)

Build the diagonal preconditioner: Curtis-Reid, then geometric mean, then Ruiz, then Pock-Chambolle
(each stage computed on the matrix scaled by the previous ones).
"""
function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At) = milp
    (; chambolle_pock_alpha, ruiz_iter, geometric_mean_iter, curtis_reid_iter) = params
    cons = ConstraintMatrix(A, At)
    prec = identity_preconditioner(cons)
    if curtis_reid_iter > 0
        prec_cr = curtis_reid_preconditioner(cons; iterations = curtis_reid_iter)
        cons = precondition(cons, prec_cr)
        prec = prec_cr * prec
    end
    if geometric_mean_iter > 0
        prec_gm = geometric_mean_preconditioner(cons; iterations = geometric_mean_iter)
        cons = precondition(cons, prec_gm)
        prec = prec_gm * prec
    end
    prec_r = ruiz_preconditioner(cons; iterations = ruiz_iter)
    cons_r = precondition(cons, prec_r)
    # a `NaN` exponent skips the Pock-Chambolle step
    prec_cp = if isnan(chambolle_pock_alpha)
        identity_preconditioner(cons_r)
    else
        chambolle_pock_preconditioner(cons_r; alpha = chambolle_pock_alpha)
    end
    return prec_cp * prec_r * prec
end

# Bound and objective rescaling

"""
    bound_objective_factors(milp; enabled = true, sum_squares = false)

Return the factors `(β, γ) = (1 / (1 + ‖b‖₂), 1 / (1 + ‖c‖₂))`. By default `bᵢ = combine(lcᵢ, ucᵢ)`
keeps the largest finite absolute bound of each constraint (the HPR-LP definition of `‖b‖`,
also used for the relative primal error). With `sum_squares = true`, `‖b‖₂²` instead sums `l²`
and `u²` over the finite bounds, counting an equality row once (the cuPDLPx definition).

Each factor is a vector with one value per instance when the quantity it is computed from is
batched, and a single number shared by the whole batch otherwise. With `enabled = false` they
are ones of the same shapes (`preprocess` then skips the rescaling altogether).
"""
function bound_objective_factors(
        milp::MILP{T}; enabled::Bool = true, sum_squares::Bool = false
    ) where {T}
    (; c, lc, uc) = milp
    b_norm = if sum_squares
        colnorm(sqrt.(bound_sum_squares.(lc, uc)))
    else
        colnorm(combine.(lc, uc))
    end
    c_norm = colnorm(c)
    β = broadcast(n -> enabled ? inv(one(T) + n) : one(T), b_norm)
    γ = broadcast(n -> enabled ? inv(one(T) + n) : one(T), c_norm)
    return β, γ
end

"""
    bound_sum_squares(l, u)

Return the contribution `l² + u²` of one constraint to `‖b‖₂²` in cuPDLPx, over its finite bounds,
with an equality constraint counted once.
"""
function bound_sum_squares(l::Number, u::Number)
    ls = ifelse(isfinite(l) & (l != u), l^2, zero(l))
    us = ifelse(isfinite(u), u^2, zero(u))
    return ls + us
end

"""
    scale_columns(x, s)

Multiply each column of `x` by the corresponding per-instance factor in `s` (a number or a vector).
"""
scale_columns(x::AbstractVecOrMat, s::Number) = x .* s
scale_columns(x::AbstractVecOrMat, s::AbstractVector) = x .* reshape(s, 1, :)

"""
    rescale_bounds_objective(milp, β, γ)

Return the problem with bounds `lc, uc, lv, uv` multiplied by `β` and objective `c` multiplied by
`γ`, recording both factors in `milp.bound_scale` and `milp.obj_scale`.

A solution `(x, y)` of `milp` corresponds to the solution `(β x, γ y)` of the rescaled problem,
whose objective is `β γ` times the original one, up to the constant `c0` which is left untouched.
"""
function rescale_bounds_objective(milp::MILP, β::BatchedNumber, γ::BatchedNumber)
    (;
        c, c0, lv, uv, A, At, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = scale_columns(c, γ),
        c0,
        lv = scale_columns(lv, β),
        uv = scale_columns(uv, β),
        A,
        At,
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

function rescale_bounds_objective(sol::PrimalDualSolution, β::BatchedNumber, γ::BatchedNumber)
    return PrimalDualSolution(scale_columns(sol.x, β), scale_columns(sol.y, γ))
end

"""
    unscale(sol, milp)

Map a solution of the preconditioned and rescaled `milp` back to the original problem.
"""
function unscale(sol::PrimalDualSolution, milp::MILP)
    (; D1, D2, bound_scale, obj_scale) = milp
    x = D2 * scale_columns(sol.x, inv.(bound_scale))
    y = D1 * scale_columns(sol.y, inv.(obj_scale))
    return PrimalDualSolution(x, y)
end
