module CoolPDLPCUDAExt

using CUDA.CUSPARSE: CuSparseMatrixCOO
using CoolPDLP: CoolPDLP

function CoolPDLP.sametype_transpose(A::CuSparseMatrixCOO)
    return CuSparseMatrixCOO(A.colInd, A.rowInd, A.nzVal, (A.dims[2], A.dims[1]), A.nnz)
end

end
