module CoolPDLPCUDAExt

using CUDA: CUDA  # not used for now
using cuSPARSE: CuSparseMatrixCOO
using CoolPDLP: CoolPDLP
using DispatchDoctor: @stable

@stable default_mode = CoolPDLP.dispatch_doctor_mode() begin

    function CoolPDLP.sametype_transpose(A::CuSparseMatrixCOO)
        return CuSparseMatrixCOO(A.colInd, A.rowInd, A.nzVal, (A.dims[2], A.dims[1]), A.nnz)
    end

end

end
