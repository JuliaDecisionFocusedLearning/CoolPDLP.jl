module CoolPDLPCUDAExt

using CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO
using CoolPDLP: CoolPDLP

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCOO(A.colInd, A.rowInd, A.nzVal, (A.dims[2], A.dims[1]), A.nnz)
end

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    At_csr = CuSparseMatrixCSR(transpose(A))
    return CuSparseMatrixCSC(At_csr)
end

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    At_csc = CuSparseMatrixCSC(transpose(A))
    return CuSparseMatrixCSR(At_csc)
end

end
