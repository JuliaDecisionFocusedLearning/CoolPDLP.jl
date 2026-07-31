"""
    Preconditioner

# Fields

$(TYPEDFIELDS)
"""
struct Preconditioner{T <: Number, Dg1 <: DiagonalScaling{T}, Dg2 <: DiagonalScaling{T}}
    "left preconditioner"
    D1::Dg1
    "right preconditioner"
    D2::Dg2
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
    A_p, At_p = precondition_matrices(A, At, prec)
    return ConstraintMatrix(A_p, At_p)
end

"""
    precondition_matrices(A, At, prec)

Apply the scalings of `prec` to a constraint matrix and its transpose.
"""
function precondition_matrices(A, At, prec::Preconditioner)
    (; D1, D2) = prec
    return (D1 * A * D2, D2 * At * D1)
end

function precondition_matrices(
        A::BatchedGPUSparseMatrixCSR, At::BatchedGPUSparseMatrixCSR, prec::Preconditioner
    )
    d1, d2 = diag(prec.D1), diag(prec.D2)
    return (scale_rows_cols(A, d1, d2), scale_rows_cols(At, d2, d1))
end

"""
    scale_rows_cols(A, dr, dc)

Scale row `i` of instance `k` of the batched matrix `A` by `dr[i, k]`, and its column `j` by `dc[j, k]`.

Only defined on the CPU, which is where preconditioning happens.
"""
function scale_rows_cols(
        A::BatchedGPUSparseMatrixCSR, dr::AbstractMatrix, dc::AbstractMatrix
    )
    (; m, n, rowptr, colval, nzval) = A
    scaled = similar(nzval)
    for k in axes(nzval, 2), i in 1:m, p in rowptr[i]:(rowptr[i + 1] - 1)
        scaled[p, k] = dr[i, k] * nzval[p, k] * dc[colval[p], k]
    end
    return BatchedGPUSparseMatrixCSR(m, n, rowptr, colval, scaled)
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
    c_p = D2 * c
    lv_p, uv_p = D2 \ lv, D2 \ uv
    A_p, At_p = precondition_matrices(A, At, prec)
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

function chambolle_pock_preconditioner(cons::ConstraintMatrix; alpha::Number)
    return diagonal_norm_preconditioner(cons; p_row = 2 - alpha, p_col = alpha)
end

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

function pdlp_preconditioner(cons::ConstraintMatrix, params::PreconditioningParameters)
    (; chambolle_pock_alpha, ruiz_iter) = params
    prec_r = ruiz_preconditioner(cons; iterations = ruiz_iter)
    cons_r = precondition(cons, prec_r)
    prec_cp = chambolle_pock_preconditioner(cons_r; alpha = chambolle_pock_alpha)
    prec = prec_r * prec_cp
    return prec
end

function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At) = milp
    if ndims(A) > 2
        return batched_preconditioner(A, At, params)
    end
    return pdlp_preconditioner(ConstraintMatrix(A, At), params)
end

"""
    batched_preconditioner(A, At, params)

Precondition each instance of a batched constraint matrix on its own, and gather the scalings into a pair of [`BatchedDiagonal`](@ref).
"""
function batched_preconditioner(A, At, params::PreconditioningParameters)
    precs = map(axes(A, 3)) do i
        # every instance goes through the plain CSC pipeline
        cons = ConstraintMatrix(
            SparseMatrixCSC(view(A, :, :, i)), SparseMatrixCSC(view(At, :, :, i))
        )
        return pdlp_preconditioner(cons, params)
    end
    return Preconditioner(
        BatchedDiagonal(stack(prec -> diag(prec.D1), precs)),
        BatchedDiagonal(stack(prec -> diag(prec.D2), precs)),
    )
end
