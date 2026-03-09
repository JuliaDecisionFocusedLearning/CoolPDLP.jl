module CoolPDLPQuadraticModelsExt

import QuadraticModels: QuadraticModel
import CoolPDLP

function CoolPDLP.MILP(qm::QuadraticModel; ignore_islp = false, kwargs...)
    ignore_islp || @assert qm.meta.islp

    return CoolPDLP.MILP(;
        c = qm.data.c,
        lv = qm.meta.lvar,
        uv = qm.meta.uvar,
        A = qm.data.A,
        lc = qm.meta.lcon,
        uc = qm.meta.ucon,
        name = qm.meta.name,
        kwargs...
    )
end

end # module
