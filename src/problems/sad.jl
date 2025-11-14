"""
    SaddlePointProblem

Represent the saddle point problem

    min_x max_y L(x, y) = cᵀx - yᵀKx + qᵀy
    s.t. x ∈ X = {l ≤ x ≤ u}
         y ∈ Y = {y[ineq_cons] ≥ 0} 

# Fields

$(TYPEDFIELDS)
"""
struct SaddlePointProblem{
        T <: Number,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
        Vb <: AbstractVector{Bool},
        P <: Preconditioner,
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
    "indicators of inequality constraints"
    ineq_cons::Vb
    "preconditioner"
    preconditioner::P

    function SaddlePointProblem(; c, q, K, Kᵀ, l, u, ineq_cons, preconditioner)
        T = Base.promote_eltype(c, q, K, Kᵀ, l, u)
        V = promote_type(
            typeof(c), typeof(q),
            typeof(l), typeof(u),
        )
        M = promote_type(typeof(K), typeof(Kᵀ))
        Vb = typeof(ineq_cons)
        P = typeof(preconditioner)
        @assert isconcretetype(T)
        @assert isconcretetype(V)
        @assert isconcretetype(M)
        @assert isconcretetype(Vb)
        return new{T, V, M, Vb, P}(c, q, K, Kᵀ, l, u, ineq_cons, preconditioner)
    end
end

"""
    SaddlePointProblem(milp::MILP)

Construct a [`SaddlePointProblem`](@ref) from a [`MILP`](@ref) as in the PDLP paper:

- `K = vcat(G, A)`
- `q = vcat(h, b)`
- `ineq_cons = (1:(m₁ + m₂)) .<= m₁`

# Fields

$(TYPEDFIELDS)
"""
function SaddlePointProblem(milp::MILP)
    (; c, G, h, A, b, l, u) = milp
    q = vcat(h, b)
    K = vcat(G, A)
    Kᵀ = SparseMatrixCSC(transpose(K))
    m₁ = length(h)
    m₂ = length(b)
    ineq_cons = similar(q, Bool)
    ineq_cons .= (1:(m₁ + m₂)) .<= m₁
    preconditioner = identity_preconditioner(K)
    return SaddlePointProblem(; c, q, K, Kᵀ, l, u, ineq_cons, preconditioner)
end

function Base.show(io::IO, sad::SaddlePointProblem)
    return print(
        io, """
        Saddle point problem
        - variables: $(nbvar(sad))
        - constraints $(nbcons(sad)) ($(nbcons_ineq(sad)) inequalities, $(nbcons_eq(sad)) equalities)
        - nonzeros: $(nnz(sad.K))"""
    )
end

Base.eltype(::SaddlePointProblem{T}) where {T} = T

nbvar(sad::SaddlePointProblem) = length(sad.c)
nbcons(sad::SaddlePointProblem) = length(sad.q)
nbcons_ineq(sad::SaddlePointProblem) = sum(sad.ineq_cons)
nbcons_eq(sad::SaddlePointProblem) = nbcons(sad) - nbcons_ineq(sad)

function apply(preconditioner::Preconditioner, sad::SaddlePointProblem)
    (; D1, D2) = preconditioner
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    K̃ = D1 * K * D2
    K̃ᵀ = D2 * Kᵀ * D1
    c̃ = D2 * c
    q̃ = D1 * q
    l̃ = D2 \ l
    ũ = D2 \ u
    return SaddlePointProblem(;
        c = c̃, q = q̃,
        K = K̃, Kᵀ = K̃ᵀ,
        l = l̃, u = ũ,
        ineq_cons,
        preconditioner = preconditioner * sad.preconditioner
    )
end
