"""
    sort_columns(A::SparseMatrixCSC)

Return a version `A_sorted` of `A` where the columns have been sorted in order of increasing number of nonzeros, along with the associated column permutation `p` such that `A_sorted == A[:, p]`
"""
function sort_columns(A::SparseMatrixCSC)
    (; colptr, rowval, nzval) = A
    col_lengths = diff(colptr)
    col_perm = sortperm(col_lengths)
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
