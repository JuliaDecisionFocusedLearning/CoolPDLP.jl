"""
    MILP

Represent a Mixed Integer Linear Program in "PDLP form":

    min cᵀx   s.t.   Gx ≥ h, Ax = b, l ≤ x ≤ u

# Fields

$(TYPEDFIELDS)
"""
struct MILP{
        T <: Number,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
        Vb <: AbstractVector{Bool},
        Vs <: AbstractVector{String},
    }
    c::V
    G::M
    h::V
    A::M
    b::V
    l::V
    u::V
    intvar::Vb
    varname::Vs

    function MILP(; c, G, h, A, b, l, u, intvar, varname)
        T = Base.promote_eltype(c, G, h, A, b, l, u)
        V = promote_type(typeof(c), typeof(h), typeof(b), typeof(l), typeof(u))
        Vb = typeof(intvar)
        Vs = typeof(varname)
        M = promote_type(typeof(G), typeof(A))
        n = length(c)
        @assert n == length(l) == length(u)
        @assert n == length(intvar) == length(varname)
        @assert n == size(G, 2) == size(A, 2)
        @assert length(h) == size(G, 1)
        @assert length(b) == size(A, 1)
        return new{T, V, M, Vb, Vs}(c, G, h, A, b, l, u, intvar, varname)
    end
end

function Base.show(io::IO, milp::MILP)
    (; c, h, b, intvar) = milp
    return print(io, "MILP with $(length(c)) variables ($(sum(intvar)) integer), $(length(h)) inequality constraints and $(length(b)) equality constraints")
end

Base.eltype(::MILP{T}) where {T} = T

nbvar(milp::MILP) = length(milp.c)

function relax(milp::MILP)
    (; c, G, h, A, b, l, u, intvar, varname) = milp
    return MILP(; c, G, h, A, b, l, u, intvar = zero(intvar), varname)
end
