"""
    AbstractProgram{T, V, M, Vb}

Abstract supertype for optimization programs in "cuPDLPx form":

    min (1/2)xᵀQx + cᵀx   s.t.   lv ≤ x ≤ uv
                                 lc ≤ A * x ≤ uc

Subtypes: [`LinearProgram`](@ref) (Q = 0) and [`QuadraticProgram`](@ref) (Q ≠ 0).
"""
abstract type AbstractProgram{
    T <: Number,
    V <: DenseVector{T},
    M <: AbstractMatrix{T},
    Vb <: DenseVector{Bool},
}
end

function _validate_program_args(; c, lv, uv, A, At, lc, uc, D1, D2, int_var, Q = nothing)
    m, n = size(A)
    if !(n == length(c) == length(lv) == length(uv) == size(D2, 1) == length(int_var))
        throw(DimensionMismatch("Variable size not consistent"))
    elseif !(m == length(lc) == length(uc) == size(D1, 2))
        throw(DimensionMismatch("Constraint size not consistent"))
    end
    if !isnothing(Q) && size(Q) != (n, n)
        throw(DimensionMismatch("Q must be $n × $n, got $(size(Q))"))
    end

    eltype_args = isnothing(Q) ? (c, lv, uv, A, At, lc, uc, D1, D2) : (c, lv, uv, A, At, Q, lc, uc, D1, D2)
    T = Base.promote_eltype(eltype_args...)
    V = promote_type(typeof(c), typeof(lv), typeof(uv), typeof(lc), typeof(uc))
    M = promote_type(typeof(A), typeof(At))
    Vb = typeof(int_var)

    if !isconcretetype(T) || !isconcretetype(V) || !isconcretetype(M) || !isconcretetype(Vb)
        throw(ArgumentError("Abstract type parameter"))
    end

    return T, V, M, Vb
end

"""
    LinearProgram

Represent a Mixed Integer Linear Program in "cuPDLPx form":

    min cᵀx   s.t.   lv ≤ x ≤ uv
                     lc ≤ A * x ≤ uc

# Constructor

    LinearProgram(;
        c, lv, uv, A, lc, uc,
        [D1, D2, int_var, var_names, dataset, name, path]
    )

# Fields

$(TYPEDFIELDS)
"""
struct LinearProgram{
        T <: Number,
        V <: DenseVector{T},
        M <: AbstractMatrix{T},
        Vb <: DenseVector{Bool},
    } <: AbstractProgram{T, V, M, Vb}
    "objective vector"
    c::V
    "variable lower bound"
    lv::V
    "variable upper bound"
    uv::V
    "constraint matrix"
    A::M
    "transposed constraint matrix"
    At::M
    "constraint lower bound"
    lc::V
    "constraint upper bound"
    uc::V
    "left preconditioner"
    D1::Diagonal{T, V}
    "right preconditioner"
    D2::Diagonal{T, V}
    "which variables must be integers"
    int_var::Vb
    "variable names"
    var_names::Vector{String}
    "source dataset"
    dataset::String
    "instance name (last part of the path)"
    name::String
    "file path the program was read from"
    path::String

    function LinearProgram(;
            c,
            lv,
            uv,
            A,
            At = convert(typeof(A), transpose(A)),
            lc,
            uc,
            D1 = Diagonal(one!(similar(lc))),
            D2 = Diagonal(one!(similar(lv))),
            int_var = zero!(similar(c, Bool)),
            var_names = map(string, eachindex(c)),
            dataset = "",
            name = "",
            path = ""
        )
        T, V, M, Vb = _validate_program_args(; c, lv, uv, A, At, lc, uc, D1, D2, int_var)

        common_backend(c, lv, uv, A, At, lc, uc, D1, D2)

        if isempty(name) && !isempty(path)
            name = splitext(splitpath(path)[end])[1]
        end

        return new{T, V, M, Vb}(
            c, lv, uv, A, At, lc, uc, D1, D2,
            int_var, var_names,
            string(dataset), string(name), string(path)
        )
    end
end

"""
    QuadraticProgram

Represent a Mixed Integer Quadratic Program in "cuPDLPx form":

    min (1/2)xᵀQx + cᵀx   s.t.   lv ≤ x ≤ uv
                                 lc ≤ A * x ≤ uc

# Constructor

    QuadraticProgram(;
        c, lv, uv, A, Q, lc, uc,
        [D1, D2, int_var, var_names, dataset, name, path]
    )

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticProgram{
        T <: Number,
        V <: DenseVector{T},
        M <: AbstractMatrix{T},
        Vb <: DenseVector{Bool},
    } <: AbstractProgram{T, V, M, Vb}
    "objective vector"
    c::V
    "variable lower bound"
    lv::V
    "variable upper bound"
    uv::V
    "constraint matrix"
    A::M
    "transposed constraint matrix"
    At::M
    "objective Hessian (symmetric positive semidefinite)"
    Q::M
    "constraint lower bound"
    lc::V
    "constraint upper bound"
    uc::V
    "left preconditioner"
    D1::Diagonal{T, V}
    "right preconditioner"
    D2::Diagonal{T, V}
    "which variables must be integers"
    int_var::Vb
    "variable names"
    var_names::Vector{String}
    "source dataset"
    dataset::String
    "instance name (last part of the path)"
    name::String
    "file path the program was read from"
    path::String

    function QuadraticProgram(;
            c,
            lv,
            uv,
            A,
            At = convert(typeof(A), transpose(A)),
            Q,
            lc,
            uc,
            D1 = Diagonal(one!(similar(lc))),
            D2 = Diagonal(one!(similar(lv))),
            int_var = zero!(similar(c, Bool)),
            var_names = map(string, eachindex(c)),
            dataset = "",
            name = "",
            path = ""
        )
        T, V, M, Vb = _validate_program_args(; c, lv, uv, A, At, lc, uc, D1, D2, int_var, Q)

        Q_M = convert(M, Q)

        common_backend(c, lv, uv, A, At, Q_M, lc, uc, D1, D2)

        if isempty(name) && !isempty(path)
            name = splitext(splitpath(path)[end])[1]
        end

        return new{T, V, M, Vb}(
            c, lv, uv, A, At, Q_M, lc, uc, D1, D2,
            int_var, var_names,
            string(dataset), string(name), string(path)
        )
    end
end

"""
    LinearProgram(qps::QPSData; kwargs...)

Construct a [`LinearProgram`](@ref) from a `QPSData` object
generated by [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl).

See also [`QuadraticProgram(::QPSData)`](@ref).
"""
function LinearProgram(qps::QPSData; kwargs...)
    return LinearProgram(;
        c = qps.c,
        lv = qps.lvar,
        uv = qps.uvar,
        A = sparse(qps.arows, qps.acols, qps.avals, length(qps.lcon), length(qps.lvar)),
        At = sparse(qps.acols, qps.arows, qps.avals, length(qps.lvar), length(qps.lcon)),
        lc = qps.lcon,
        uc = qps.ucon,
        D1 = Diagonal(ones(length(qps.lcon))),
        D2 = Diagonal(ones(length(qps.lvar))),
        int_var = convert(Vector{Bool}, (qps.vartypes .== VTYPE_Binary) .| (qps.vartypes .== VTYPE_Integer)),
        var_names = qps.varnames,
        kwargs...
    )
end

"""
    QuadraticProgram(qps::QPSData; kwargs...)

Construct a [`QuadraticProgram`](@ref) from a `QPSData` object
generated by [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl).

See also [`LinearProgram(::QPSData)`](@ref).
"""
function QuadraticProgram(qps::QPSData; kwargs...)
    n = length(qps.lvar)
    Q_upper = sparse(qps.qrows, qps.qcols, qps.qvals, n, n)
    Q = Q_upper + Q_upper' - Diagonal(diag(Q_upper))
    return QuadraticProgram(;
        c = qps.c,
        lv = qps.lvar,
        uv = qps.uvar,
        A = sparse(qps.arows, qps.acols, qps.avals, length(qps.lcon), length(qps.lvar)),
        At = sparse(qps.acols, qps.arows, qps.avals, length(qps.lvar), length(qps.lcon)),
        Q,
        lc = qps.lcon,
        uc = qps.ucon,
        D1 = Diagonal(ones(length(qps.lcon))),
        D2 = Diagonal(ones(length(qps.lvar))),
        int_var = convert(Vector{Bool}, (qps.vartypes .== VTYPE_Binary) .| (qps.vartypes .== VTYPE_Integer)),
        var_names = qps.varnames,
        kwargs...
    )
end

# Show methods

function Base.show(io::IO, milp::AbstractProgram{T, V, M}) where {T, V, M}
    type_name = nameof(typeof(milp))
    quadratic_line = milp isa QuadraticProgram ? "\n        - quadratic: true" : ""
    return print(
        io, """
        $type_name instance $(milp.name) from dataset $(milp.dataset):
        - types:
          - values $T
          - vectors $V
          - matrices $M
        - variables: $(nbvar(milp))
          - $(nbvar_cont(milp)) continuous
          - $(nbvar_int(milp)) integer
        - constraints: $(nbcons(milp))
          - $(nbcons_ineq(milp)) inequalities
          - $(nbcons_eq(milp)) equalities
        - nonzeros: $(mynnz(milp.A))$quadratic_line"""
    )
end

KernelAbstractions.get_backend(milp::AbstractProgram) = get_backend(milp.c)


get_Q(::LinearProgram) = nothing
get_Q(milp::QuadraticProgram) = milp.Q

rebuild(::LinearProgram; Q::Nothing = nothing, kwargs...) = LinearProgram(; kwargs...)
rebuild(::QuadraticProgram; kwargs...) = QuadraticProgram(; kwargs...)

"""
    nbvar(milp)

Return the number of variables in `milp`.
"""
nbvar(milp::AbstractProgram) = length(milp.c)

"""
    nbvar_int(milp)

Return the number of integer variables in `milp`.
"""
nbvar_int(milp::AbstractProgram) = sum(milp.int_var)

"""
    nbvar_cont(milp)

Return the number of continuous variables in `milp`.
"""
nbvar_cont(milp::AbstractProgram) = nbvar(milp) - nbvar_int(milp)

"""
    nbcons(milp)

Return the number of constraints in `milp`, not including variable bounds or integrality requirements.
"""
nbcons(milp::AbstractProgram) = size(milp.A, 1)

"""
    nbcons_eq(milp)

Return the number of equality constraints in `milp`.
"""
nbcons_eq(milp::AbstractProgram) = mapreduce((l, u) -> (l == u), +, milp.lc, milp.uc)

"""
    nbcons_ineq(milp)

Return the number of inequality constraints in `milp`, not including variable bounds.
"""
nbcons_ineq(milp::AbstractProgram) = nbcons(milp) - nbcons_eq(milp)
