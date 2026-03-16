module CoolPDLPCUDAExt

using CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO
using CoolPDLP: CoolPDLP

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCOO(A.colInd, A.rowInd, A.nzVal, (A.dims[2], A.dims[1]), A.nnz)
end

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCSC(transpose(A))
end

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCSR(transpose(A))
end

end
