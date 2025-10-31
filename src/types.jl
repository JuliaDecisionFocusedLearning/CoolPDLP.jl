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


"""
    nbvar(milp)

Return the number of variables in `milp`.
"""
nbvar(milp::MILP) = length(milp.c)

"""
    nbcons(milp)

Return the number of constraints in `milp`, not including variable bounds or integrality requirements.
"""
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
        I <: Integer,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
    }
    c::V
    q::V
    K::M
    Kᵀ::M
    l::V
    u::V
    m₁::I
    m₂::I

    function SaddlePointProblem(; c, q, K, Kᵀ, l, u, m₁, m₂)
        T = Base.promote_eltype(c, q, K, Kᵀ, l, u)
        I = promote_type(typeof(m₁), typeof(m₂))
        V = promote_type(typeof(c), typeof(q), typeof(l), typeof(u))
        M = promote_type(typeof(K), typeof(Kᵀ))
        return new{T, I, V, M}(c, q, K, Kᵀ, l, u, m₁, m₂)
    end
end

function SaddlePointProblem(milp::MILP)
    (; c, G, h, A, b, l, u) = milp
    q = vcat(h, b)
    K = vcat(G, A)
    Kᵀ = convert(typeof(K), transpose(K))
    Kd = DeviceSparseMatrixCSR(K)
    Kᵀd = DeviceSparseMatrixCSR(Kᵀ)
    m₁ = length(h)
    m₂ = length(b)
    return SaddlePointProblem(; c, q, K = Kd, Kᵀ = Kᵀd, l, u, m₁, m₂)
end

function Adapt.adapt_structure(to, problem::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = problem
    return SaddlePointProblem(;
        c = adapt(to, c),
        q = adapt(to, q),
        K = adapt(to, K),
        Kᵀ = adapt(to, Kᵀ),
        l = adapt(to, l),
        u = adapt(to, u),
        m₁ = m₁,
        m₂ = m₂
    )
end

function change_eltype(::Type{T}, problem::SaddlePointProblem) where {T}
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = problem
    return SaddlePointProblem(;
        c = change_eltype(T, c),
        q = change_eltype(T, q),
        K = change_eltype(T, K),
        Kᵀ = change_eltype(T, Kᵀ),
        l = change_eltype(T, l),
        u = change_eltype(T, u),
        m₁ = m₁,
        m₂ = m₂
    )
end

struct PrimalDualVariable{T <: Number, V <: AbstractVector{T}}
    x::V
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

function default_init(problem::SaddlePointProblem)
    return PrimalDualVariable(zero(problem.c), zero(problem.q))
end
