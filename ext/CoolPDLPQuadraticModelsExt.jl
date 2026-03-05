module CoolPDLPQuadraticModelsExt

import SparseArrays: SparseMatrixCSC
import QuadraticModels: AbstractQuadraticModel
import CoolPDLP

function CoolPDLP.MILP(qm::AbstractQuadraticModel; ignore_islp = false, kwargs...)
    ignore_islp || @assert qm.meta.islp

    return CoolPDLP.MILP(;
        c = qm.data.c,
        lv = qm.meta.lvar,
        uv = qm.meta.uvar,
        A = SparseMatrixCSC(qm.data.A),
        lc = qm.meta.lcon,
        uc = qm.meta.ucon,
        name = qm.meta.name,
        kwargs...
    )
end

end # module
