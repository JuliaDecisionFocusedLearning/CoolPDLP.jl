abstract type AbstractProblem end

"""
    nbvar(problem)

Return the number of variables in `problem`.
"""
function nbvar end

"""
    nbvar_int(problem)

Return the number of integer variables in `problem`.
"""
function nbvar_int end

"""
    nbvar_cont(problem)

Return the number of integer variables in `problem`.
"""
function nbvar_cont end

"""
    nbcons(problem)

Return the number of constraints in `problem`, not including variable bounds or integrality requirements.
"""
function nbcons end

"""
    nbcons_eq(problem)

Return the number of equality constraints in `problem`.
"""
function nbcons_eq end

"""
    nbcons_ineq(problem)

Return the number of inequality constraints in `problem`, not including variable bounds.
"""
function nbcons_ineq end

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
        M = promote_type(typeof(G), typeof(A))
        Vb = typeof(intvar)
        Vs = typeof(varname)
        @assert isconcretetype(T)
        @assert isconcretetype(V)
        @assert isconcretetype(M)

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
         y ∈ Y = {y[ineq_cons] ≥ 0} 

# Fields

$(TYPEDFIELDS)
"""
struct SaddlePointProblem{
        T <: Number,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
        Vb <: AbstractVector{Bool},
        Dv <: Diagonal{T},
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
    "variable lower bound without infinite values"
    l_noinf::V
    "variable upped bound without infinite values"
    u_noinf::V
    "indicators of inequality constraints"
    ineq_cons::Vb
    "left preconditioner"
    D1::Dv
    "right preconditioner"
    D2::Dv

    function SaddlePointProblem(; c, q, K, Kᵀ, l, u, l_noinf, u_noinf, ineq_cons, D1, D2)
        T = Base.promote_eltype(c, q, K, Kᵀ, l, u, D1, D2)
        V = promote_type(
            typeof(c), typeof(q),
            typeof(l), typeof(u),
            typeof(l_noinf), typeof(u_noinf)
        )
        M = promote_type(typeof(K), typeof(Kᵀ))
        Vb = typeof(ineq_cons)
        Dv = promote_type(typeof(D1), typeof(D2))
        @assert isconcretetype(T)
        @assert isconcretetype(V)
        @assert isconcretetype(M)
        @assert isconcretetype(Vb)
        @assert isconcretetype(Dv)
        return new{T, V, M, Vb, Dv}(c, q, K, Kᵀ, l, u, l_noinf, u_noinf, ineq_cons, D1, D2)
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
function SaddlePointProblem(milp::MILP{T}) where {T}
    (; c, G, h, A, b, l, u) = milp
    q = vcat(h, b)
    K = vcat(G, A)
    Kᵀ = convert(typeof(K), transpose(K))
    l_noinf = max.(nextfloat(typemin(T)), l)
    u_noinf = min.(prevfloat(typemax(T)), u)
    m₁ = length(h)
    m₂ = length(b)
    ineq_cons = similar(q, Bool)
    ineq_cons .= (1:(m₁ + m₂)) .<= m₁
    d1 = similar(q)
    d2 = similar(c)
    fill!(d1, one(T))
    fill!(d2, one(T))
    D1 = Diagonal(d1)
    D2 = Diagonal(d2)
    return SaddlePointProblem(; c, q, K, Kᵀ, l, u, l_noinf, u_noinf, ineq_cons, D1, D2)
end

function Base.show(io::IO, sad::SaddlePointProblem)
    return print(io, "Saddle point problem with $(nbvar(sad)) variables, $(nbcons_ineq(sad)) inequality constraints and $(nbcons_eq(sad)) equality constraints (total of $(nnz(sad.K)) nonzero coefficients)")
end

Base.eltype(::SaddlePointProblem{T}) where {T} = T

nbvar(sad::SaddlePointProblem) = length(sad.c)
nbcons(sad::SaddlePointProblem) = length(sad.q)
nbcons_ineq(sad::SaddlePointProblem) = sum(sad.ineq_cons)
nbcons_eq(sad::SaddlePointProblem) = nbcons(sad) - nbcons_ineq(sad)

"""
    TerminationReason

Enum type listing possible reasons for algorithm termination:

- `CONVERGENCE`
- `TIME`
- `ITERATIONS`
- `STILL_RUNNING`
"""
@enum TerminationReason CONVERGENCE TIME ITERATIONS STILL_RUNNING

"""
    AbstractState

Algorithm state supertype.

!!! warning
    Work in progress.

# Required fields

- `x`, `y`
- `x_scratch1`, `x_scratch2`, `x_scratch3`, `y_scratch`
- `time_elapsed`
- `kkt_passes`
- `relative_error`
- `termination_reason`
"""
abstract type AbstractState{T <: Number, V <: AbstractVector{T}} end

"""
    AbstractParameters

Algorithm parameter supertype.

!!! warning
    Work in progress.

# Required fields

- `termination_reltol`
- `time_limit`
- `max_kkt_passes`
- `record_error_history`
"""
abstract type AbstractParameters{T <: Number} end


"""
    KKTErrors

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct KKTErrors{T <: Number}
    err_primal::T
    err_dual::T
    err_gap::T
    err_primal_scale::T
    err_dual_scale::T
    err_gap_scale::T
    max_rel_err::T
    weighted_aggregate_err::T
end

function Base.show(io::IO, err::KKTErrors)
    return print(io, "KKT errors: primal $(err.err_primal) (scale $(err.err_primal_scale)), dual $(err.err_dual) (scale $(err.err_dual_scale)), gap $(err.err_gap) (scale $(err.err_gap_scale))")
end

function KKTErrors(::Type{T}) where {T}
    return KKTErrors(
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
    )
end

"""
    PrimalDualSolution

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PrimalDualSolution{T <: Number, V <: AbstractVector{T}}
    const x::V
    const y::V
    const Kx::V
    const Kᵀy::V
    const λ::V
    const λ⁺::V
    const λ⁻::V
    err::KKTErrors{T}
end

Base.eltype(::PrimalDualSolution{T}) where {T} = T

function PrimalDualSolution(
        sad::SaddlePointProblem{T, V},
        x::V = zero(sad.c),
        y::V = zero(sad.q)
    ) where {T, V}
    (; c, K, Kᵀ, l, u) = sad
    Kx = K * x
    Kᵀy = Kᵀ * y
    λ = proj_λ.(c - Kᵀy, l, u)
    λ⁺ = positive_part.(λ)
    λ⁻ = negative_part.(λ)
    err = KKTErrors(T)
    return PrimalDualSolution(; x, y, Kx, Kᵀy, λ, λ⁺, λ⁻, err)
end

function Base.copy(z::PrimalDualSolution)
    return PrimalDualSolution(
        copy(z.x), copy(z.y),
        copy(z.Kx), copy(z.Kᵀy),
        copy(z.λ), copy(z.λ⁺), copy(z.λ⁻),
        z.err
    )
end

function Base.copyto!(z1::PrimalDualSolution, z2::PrimalDualSolution)
    copyto!(z1.x, z2.x)
    copyto!(z1.y, z2.y)
    copyto!(z1.Kx, z2.Kx)
    copyto!(z1.Kᵀy, z2.Kᵀy)
    copyto!(z1.λ, z2.λ)
    copyto!(z1.λ⁺, z2.λ⁺)
    copyto!(z1.λ⁻, z2.λ⁻)
    z1.err = z2.err
    return z1
end

function weighted_sum!(
        z1::PrimalDualSolution{T}, z2::PrimalDualSolution{T},
        a1::T, a2::T,
    ) where {T}
    @. z1.x = a1 * z1.x + a2 * z2.x
    @. z1.y = a1 * z1.y + a2 * z2.y
    @. z1.Kx = a1 * z1.Kx + a2 * z2.Kx
    @. z1.Kᵀy = a1 * z1.Kᵀy + a2 * z2.Kᵀy
    @. z1.λ = a1 * z1.λ + a2 * z2.λ
    @. z1.λ⁺ = positive_part(z1.λ)
    @. z1.λ⁻ = negative_part(z1.λ)
    z1.err = KKTErrors(T)
    return z1
end
