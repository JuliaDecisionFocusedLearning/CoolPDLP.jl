function increasing_column_order(A::SparseMatrixCSC)
    col_lengths = diff(A.colptr)
    return sortperm(col_lengths)
end

function permute_rows_columns(
        A::SparseMatrixCSC; perm_col::Vector{Int}, perm_row::Vector{Int}
    )
    (I, J, V) = findnz(A)
    (m, n) = size(A)
    invperm_row = invperm(perm_row)
    invperm_col = invperm(perm_col)
    return sparse(invperm_row[I], invperm_col[J], V, m, n)
end

"""
    sort_rows_columns(milp)

Return a new `MILP` where the constraint matrix has been permuted by order of increasing column and row density.
"""
function sort_rows_columns(milp::MILP)
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp

    perm_var = increasing_column_order(A)
    perm_cons = increasing_column_order(At)

    return MILP(;
        c = c[perm_var],
        lv = lv[perm_var],
        uv = uv[perm_var],
        A = permute_rows_columns(A; perm_col = perm_var, perm_row = perm_cons),
        At = permute_rows_columns(At; perm_col = perm_cons, perm_row = perm_var),
        lc = lc[perm_cons],
        uc = uc[perm_cons],
        D1 = Diagonal(diag(D1)[perm_cons]),
        D2 = Diagonal(diag(D2)[perm_var]),
        int_var = int_var[perm_var],
        var_names = var_names[perm_var],
        dataset,
        name,
        path
    )
end
