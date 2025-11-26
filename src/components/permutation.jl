function increasing_column_order(A::SparseMatrixCSC)
    col_lengths = diff(A.colptr)
    return sortperm(col_lengths)
end

function permute_columns(A::SparseMatrixCSC, col_perm::Vector{Int})
    (; colptr, rowval, nzval) = A
    new_colptr = similar(colptr)
    new_rowval = similar(rowval)
    new_nzval = similar(nzval)
    k = 1
    for (new_j, j) in enumerate(col_perm)
        new_colptr[new_j] = k
        for l in colptr[j]:(colptr[j + 1] - 1)
            new_rowval[k] = rowval[l]
            new_nzval[k] = nzval[l]
            k += 1
        end
    end
    new_colptr[end] = nnz(A) + 1
    return SparseMatrixCSC(A.m, A.n, new_colptr, new_rowval, new_nzval), col_perm
end

function permute_rows(A::SparseMatrixCSC, row_perm::Vector{Int})
    At = sparse(transpose(A))
    At_sorted_col = permute_columns(At, row_perm)
    return sparse(transpose(At_sorted_col))
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
        A = permute_rows(permute_columns(A, perm_var), perm_cons),
        At = permute_rows(permute_columns(At, perm_cons), perm_var),
        lc = lc[perm_cons],
        uc = uc[perm_cons],
        D1 = Diagonal(D1.diag[perm_cons]),
        D2 = Diagonal(D2.diag[perm_var]),
        int_var = int_var[perm_var],
        var_names = var_names[perm_var],
        dataset,
        name,
        path
    )
end
