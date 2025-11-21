"""
    MILP

Represent a Mixed Integer Linear Program in "PDLP form":

    min cᵀx   s.t.   Gx ≥ h, Ax = b, l ≤ x ≤ u

# Fields

$(TYPEDFIELDS)
"""
struct MILP <: AbstractProblem
    "objective vector"
    c::Vector{Float64}
    "inequality constraint matrix"
    G::SparseMatrixCSC{Float64, Int}
    "inequality constraint right-hand side"
    h::Vector{Float64}
    "equality constraint matrix"
    A::SparseMatrixCSC{Float64, Int}
    "equality constraint right-hand side"
    b::Vector{Float64}
    "variable lower bound"
    l::Vector{Float64}
    "variable upper bound"
    u::Vector{Float64}
    "specify which variables must be integers"
    intvar::Vector{Bool}
    "list of variable names"
    varname::Vector{String}
    "source dataset"
    dataset::String
    "instance name (last part of the path)"
    name::String
    "file path where the MILP was read from"
    path::String
end

function MILP(;
        c,
        G,
        h,
        A,
        b,
        l,
        u,
        intvar = fill(false, length(c)),
        varname = map(string, eachindex(c)),
        dataset = "",
        name = "",
        path = "",
    )
    n = length(c)
    m₁ = length(h)
    m₂ = length(b)
    @assert n == length(l) == length(u)
    @assert n == length(intvar) == length(varname)
    @assert n == size(G, 2) == size(A, 2)
    @assert m₁ == size(G, 1)
    @assert m₂ == size(A, 1)

    @assert all(isfinite, c)
    @assert all(isfinite, h)
    @assert all(isfinite, b)

    if isempty(name) && !isempty(path)
        name = splitext(splitpath(path)[end])[1]
    end

    return MILP(c, G, h, A, b, l, u, intvar, varname, dataset, name, path)
end

function Base.show(io::IO, milp::MILP)
    return print(
        io,
        """
        MILP instance $(milp.name)
        - variables: $(nbvar(milp)) ($(nbvar_cont(milp)) continuous, $(nbvar_int(milp)) integer)
        - constraints: $(nbcons(milp)) ($(nbcons_ineq(milp)) inequalities, $(nbcons_eq(milp)) equalities)
        - nonzeros: $(nnz(milp.G) + nnz(milp.A))""",
    )
end

nbvar(milp::MILP) = length(milp.c)
nbvar_int(milp::MILP) = sum(milp.intvar)
nbvar_cont(milp::MILP) = nbvar(milp) - nbvar_int(milp)

nbcons(milp::MILP) = nbcons_eq(milp) + nbcons_ineq(milp)
nbcons_eq(milp::MILP) = length(milp.b)
nbcons_ineq(milp::MILP) = length(milp.h)

"""
    relax(milp)

Return a new `MILP` identical to `milp` but without integrality requirements.
"""
relax(milp::MILP) = @set milp.intvar = zero(milp.intvar)
