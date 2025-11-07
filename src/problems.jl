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
        Tv <: Number,
        V <: AbstractVector{Tv},
        M <: AbstractMatrix{Tv},
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
        Tv = Base.promote_eltype(c, G, h, A, b, l, u)
        V = promote_type(typeof(c), typeof(h), typeof(b), typeof(l), typeof(u))
        M = promote_type(typeof(G), typeof(A))
        Vb = typeof(intvar)
        Vs = typeof(varname)
        @assert isconcretetype(Tv)
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

        return new{Tv, V, M, Vb, Vs}(c, G, h, A, b, l, u, intvar, varname)
    end
end

function Base.show(io::IO, milp::MILP)
    return print(io, "MILP with $(nbvar(milp)) variables ($(nbvar_cont(milp)) continuous, $(nbvar_int(milp)) integer), $(nbcons_ineq(milp)) inequality constraints and $(nbcons_eq(milp)) equality constraints (total of $(nnz(milp.G) + nnz(milp.A)) nonzero coefficients)")
end

Base.eltype(::MILP{Tv}) where {Tv} = Tv

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
        Tv <: Number,
        V <: AbstractVector{Tv},
        M <: AbstractMatrix{Tv},
        Vb <: AbstractVector{Bool},
        Dv <: Diagonal{Tv},
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
    "left preconditioner"
    D1::Dv
    "right preconditioner"
    D2::Dv

    function SaddlePointProblem(; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2)
        Tv = Base.promote_eltype(c, q, K, Kᵀ, l, u, D1, D2)
        V = promote_type(
            typeof(c), typeof(q),
            typeof(l), typeof(u),
        )
        M = promote_type(typeof(K), typeof(Kᵀ))
        Vb = typeof(ineq_cons)
        Dv = promote_type(typeof(D1), typeof(D2))
        @assert isconcretetype(Tv)
        @assert isconcretetype(V)
        @assert isconcretetype(M)
        @assert isconcretetype(Vb)
        @assert isconcretetype(Dv)
        return new{Tv, V, M, Vb, Dv}(c, q, K, Kᵀ, l, u, ineq_cons, D1, D2)
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
function SaddlePointProblem(milp::MILP{Tv}) where {Tv}
    (; c, G, h, A, b, l, u) = milp
    q = vcat(h, b)
    K = vcat(G, A)
    Kᵀ = convert(typeof(K), transpose(K))
    m₁ = length(h)
    m₂ = length(b)
    ineq_cons = similar(q, Bool)
    ineq_cons .= (1:(m₁ + m₂)) .<= m₁
    d1 = similar(q)
    d2 = similar(c)
    fill!(d1, one(Tv))
    fill!(d2, one(Tv))
    D1 = Diagonal(d1)
    D2 = Diagonal(d2)
    return SaddlePointProblem(; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2)
end

function Base.show(io::IO, sad::SaddlePointProblem)
    return print(io, "Saddle point problem with $(nbvar(sad)) variables, $(nbcons_ineq(sad)) inequality constraints and $(nbcons_eq(sad)) equality constraints (total of $(nnz(sad.K)) nonzero coefficients)")
end

Base.eltype(::SaddlePointProblem{Tv}) where {Tv} = Tv

nbvar(sad::SaddlePointProblem) = length(sad.c)
nbcons(sad::SaddlePointProblem) = length(sad.q)
nbcons_ineq(sad::SaddlePointProblem) = sum(sad.ineq_cons)
nbcons_eq(sad::SaddlePointProblem) = nbcons(sad) - nbcons_ineq(sad)

"""
    PrimalDualVariable

Represent a couple of primal and dual variables.

# Fields

$(TYPEDFIELDS)
"""
struct PrimalDualVariable{Tv <: Number, V <: AbstractVector{Tv}}
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
- `elapsed`
- `kkt_passes`
- `relative_error`
- `termination_reason`
"""
abstract type AbstractState{Tv, V} end

function Base.show(io::IO, state::AbstractState)
    (; elapsed, kkt_passes, relative_error, termination_reason) = state
    return print(
        io,
        @sprintf(
            "%s with termination reason %s: %.2e relative KKT error after %g seconds elapsed and %s KKT passes",
            nameof(typeof(state)),
            termination_reason,
            relative_error,
            elapsed,
            kkt_passes,
        )
    )
end

"""
    AbstractParameters

Algorithm parameter supertype.

!!! warning
    Work in progress.

# Required fields

- `tol_termination`
- `time_limit`
- `max_kkt_passes`
- `record_error_history`
"""
abstract type AbstractParameters{Tv} end
