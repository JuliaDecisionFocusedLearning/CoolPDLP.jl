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
        c, lv, uv, A, At, lc, uc,
        int_var, var_names, dataset, name, path,
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
        lv = lv_p,
        uv = uv_p,
        A = A_p,
        At = At_p,
        lc = lc_p,
        uc = uc_p,
        D1 = new_prec.D1,
        D2 = new_prec.D2,
        int_var,
        var_names,
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
column `j` is `(sum_i |K[i,j]|^(2-α))^{-1/2}` and the constraint-side factor for
row `i` is `(sum_j |K[i,j]|^α)^{-1/2}`. With these, `‖Σ^{1/2} K T^{1/2}‖ ≤ 1`
where `T, Σ` are the diagonals built from those factors.
"""
function chambolle_pock_preconditioner(cons::ConstraintMatrix{T}; alpha::Number) where {T}
    (; A, At) = cons
    α_col = T(2 - alpha)
    α_row = T(alpha)
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
    PreconditioningParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PreconditioningParameters{T}
    "norm parameter in the Chambolle-pock preconditioner"
    chambolle_pock_alpha::T
    "iteration parameter in the Ruiz preconditioner"
    ruiz_iter::Int
end

function Base.show(io::IO, params::PreconditioningParameters)
    (; chambolle_pock_alpha, ruiz_iter) = params
    return print(io, "PreconditioningParameters: chambolle_pock_alpha=$chambolle_pock_alpha, ruiz_iter=$ruiz_iter")
end

function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At) = milp
    (; chambolle_pock_alpha, ruiz_iter) = params
    cons = ConstraintMatrix(A, At)
    prec_r = ruiz_preconditioner(cons; iterations = ruiz_iter)
    cons_r = precondition(cons, prec_r)
    prec_cp = chambolle_pock_preconditioner(cons_r; alpha = chambolle_pock_alpha)
    prec = prec_r * prec_cp
    return prec
end
