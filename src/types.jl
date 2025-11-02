abstract type AbstractProblem end

"""
    nbvar(problem)

Return the number of variables in `problem`.
"""
function nbvar end

"""
    nbcons(problem)

Return the number of constraints in `problem`, not including variable bounds or integrality requirements.
"""
function nbcons end

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
    } <: AbstractProblem
    "objective vector"
    c::V
    "inequality constraint matrix"
    G::M
    "inequality constraint right-hand side"
    h::V
    "equality constraint matrix"
    A::M
    "equality constraint right-hand side"
    b::V
    "variable lower bound"
    l::V
    "variable upper bound"
    u::V
    "specify which variables must be integers"
    intvar::Vb
    "list of variable names"
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
    return print(io, "MILP with $(nbvar(milp)) variables ($(nbvar_cont(milp)) continuous, $(nbvar_int(milp)) integer), $(nbcons_ineq(milp)) inequality constraints and $(nbcons_eq(milp)) equality constraints (total of $(nnz(milp.G) + nnz(milp.A)) nonzero coefficients)")
end

Base.eltype(::MILP{T}) where {T} = T

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
function relax(milp::MILP)
    (; c, G, h, A, b, l, u, intvar, varname) = milp
    return MILP(; c, G, h, A, b, l, u, intvar = zero(intvar), varname)
end


"""
    SaddlePointProblem

Represent the saddle point problem

    min_x max_y L(x, y) = cᵀx - yᵀKx + qᵀy
    s.t. x ∈ X = {l ≤ x ≤ u}
         y ∈ Y = {y[1:m₁] ≥ 0} 

# Fields

$(TYPEDFIELDS)
"""
struct SaddlePointProblem{
        T <: Number,
        Ti <: Integer,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
    } <: AbstractProblem
    "objective vector"
    c::V
    "constraint right-hand side"
    q::V
    "constraint matrix"
    K::M
    "transposed constraint matrix"
    Kᵀ::M
    "variable lower bound"
    l::V
    "variable upped bound"
    u::V
    "number of inequality constraints"
    m₁::Ti
    "number of equality constraints"
    m₂::Ti

    function SaddlePointProblem(; c, q, K, Kᵀ, l, u, m₁, m₂)
        T = Base.promote_eltype(c, q, K, Kᵀ, l, u)
        Ti = promote_type(typeof(m₁), typeof(m₂))
        V = promote_type(typeof(c), typeof(q), typeof(l), typeof(u))
        M = promote_type(typeof(K), typeof(Kᵀ))
        return new{T, Ti, V, M}(c, q, K, Kᵀ, l, u, m₁, m₂)
    end
end

"""
    SaddlePointProblem(milp::MILP)

Construct a [`SaddlePointProblem`](@ref) from a [`MILP`](@ref) as in the PDLP paper:

- `K = vcat(G, A)`
- `q = vcat(h, b)`
"""
function SaddlePointProblem(milp::MILP)
    (; c, G, h, A, b, l, u) = milp
    q = vcat(h, b)
    K = vcat(G, A)
    Kᵀ = convert(typeof(K), transpose(K))
    m₁ = length(h)
    m₂ = length(b)
    return SaddlePointProblem(; c, q, K, Kᵀ, l, u, m₁, m₂)
end

function Base.show(io::IO, sad::SaddlePointProblem)
    return print(io, "Saddle point problem with $(nbvar(sad)) variables, $(nbcons_ineq(sad)) inequality constraints and $(nbcons_eq(sad)) equality constraints (total of $(nnz(sad.K)) nonzero coefficients)")
end

Base.eltype(::SaddlePointProblem{T}) where {T} = T

nbvar(sad::SaddlePointProblem) = length(sad.c)
nbcons(sad::SaddlePointProblem) = nbcons_eq(sad) + nbcons_ineq(sad)
nbcons_eq(sad::SaddlePointProblem) = sad.m₂
nbcons_ineq(sad::SaddlePointProblem) = sad.m₁

"""
    PrimalDualVariable

Represent a couple of primal and dual variables.

# Fields

$(TYPEDFIELDS)
"""
struct PrimalDualVariable{T <: Number, V <: AbstractVector{T}}
    "primal variable"
    x::V
    "dual variable"
    y::V
end

Base.copy(z::PrimalDualVariable) = PrimalDualVariable(copy(z.x), copy(z.y))
Base.zero(z::PrimalDualVariable) = PrimalDualVariable(zero(z.x), zero(z.y))

function Base.copyto!(z1::PrimalDualVariable, z2::PrimalDualVariable)
    copyto!(z1.x, z2.x)
    copyto!(z1.y, z2.y)
    return nothing
end

function Base.:*(α::T, z::PrimalDualVariable{T}) where {T}
    return PrimalDualVariable(α * z.x, α * z.y)
end

function Base.:+(z1::PrimalDualVariable{T}, z2::PrimalDualVariable{T}) where {T}
    return PrimalDualVariable(z1.x + z2.x, z1.y + z2.y)
end

function default_init(sad::SaddlePointProblem)
    return PrimalDualVariable(zero(sad.c), zero(sad.q))
end
