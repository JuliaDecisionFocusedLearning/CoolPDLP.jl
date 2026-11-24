"""
    CoolPDLPGenericLinearAlgebraExt

Empty on purpose: its presence tells [`CoolPDLP.spectral_norm`](@ref) that GenericLinearAlgebra
is loaded, so the small bidiagonal SVD can run in the precision of the matrix instead of `Float64`.
"""
module CoolPDLPGenericLinearAlgebraExt

import GenericLinearAlgebra

end
