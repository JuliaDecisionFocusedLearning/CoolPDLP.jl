module CoolPDLPGPUArraysExt

using GPUArrays: AbstractGPUSparseMatrix, sparse_array_type
using CoolPDLP: CoolPDLP

function CoolPDLP.sametype_transpose(A::AbstractGPUSparseMatrix)
    return sparse_array_type(A)(transpose(A))
end

end
