function precondition(A::AbstractMatrix, At::AbstractMatrix, D1::Diagonal, D2::Diagonal)
    A_precond = D1 * A * D2
    At_precond = D2 * At * D1
    return A_precond, At_precond
end

function identity_preconditioner(
        A::AbstractMatrix, At::AbstractMatrix
    )
    d1 = ones(size(A, 1))
    d2 = ones(size(A, 2))
    return Diagonal(d1), Diagonal(d2)
end

function diagonal_norm_preconditioner(
        A::SparseMatrixCSC{T}, At::SparseMatrixCSC{T};
        p_row::Number, p_col::Number
    ) where {T}
    col_norms = map(j -> column_norm(A, j, p_col), axes(A, 2))
    row_norms = map(i -> column_norm(At, i, p_row), axes(A, 1))
    d1 = map(rn -> iszero(rn) ? one(T) : inv(sqrt(rn)), row_norms)
    d2 = map(cn -> iszero(cn) ? one(T) : inv(sqrt(cn)), col_norms)
    return Diagonal(d1), Diagonal(d2)
end

function chambolle_pock_preconditioner(A, At; alpha::Number)
    return diagonal_norm_preconditioner(A, At; p_row = 2 - alpha, p_col = alpha)
end

function ruiz_preconditioner(A, At; iterations::Integer)
    D1, D2 = identity_preconditioner(A, At)
    for _ in 1:iterations
        D1_next, D2_next = diagonal_norm_preconditioner(A, At; p_col = Inf, p_row = Inf)
        A, At = precondition(A, At, D1, D2)
        D1 = D1_next * D1
        D2 = D2 * D2_next
    end
    return D1, D2
end
