module CoolPDLPGPUArraysExt

using GPUArrays: AbstractGPUSparseMatrix, sparse_array_type
using CoolPDLP: CoolPDLP
using DispatchDoctor: @stable

@stable default_mode = CoolPDLP.dispatch_doctor_mode() begin

    function CoolPDLP.sametype_transpose(A::AbstractGPUSparseMatrix)
        return sparse_array_type(A)(transpose(A))
    end

end

end
