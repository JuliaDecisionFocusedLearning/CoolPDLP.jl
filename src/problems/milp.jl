"""
    MILP

Represent a Mixed Integer Linear Program in "cuPDLPx form":

    min cᵀx   s.t.   lv ≤ x ≤ uv
                     lc ≤ A * x ≤ uc

# Fields

$(TYPEDFIELDS)
"""
struct MILP{
        T <: Number,
        V <: AbstractVector{T},
        M <: AbstractMatrix{T},
        Vb <: AbstractVector{Bool},
    }
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
    "file path where the MILP was read from"
    path::String

    function MILP(;
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
            dataset = "Unknown dataset",
            name = "Unknown name",
            path = "Unknown path"
        )
        m, n = size(A)
        if !(n == length(c) == length(lv) == length(uv) == size(D2, 1) == length(int_var) == length(var_names))
            throw(DimensionMismatch("Variable size not consistent"))
        elseif !(m == length(lc) == length(uc) == size(D1, 2))
            throw(DimensionMismatch("Constraint size not consistent"))
        end

        T = Base.promote_eltype(c, lv, uv, A, At, lc, uc, D1, D2)
        V = promote_type(typeof(c), typeof(lv), typeof(uv), typeof(lc), typeof(uc))
        M = promote_type(typeof(A), typeof(At))
        Vb = typeof(int_var)

        if !isconcretetype(T) || !isconcretetype(V) || ! isconcretetype(M) || !isconcretetype(Vb)
            throw(ArgumentError("Abstract type parameter"))
        end

        backends = common_backend(c, lv, uv, A, At, lc, uc, D1, D2)

        if isempty(name) && !isempty(path)
            name = splitext(splitpath(path)[end])[1]
        end

        return new{T, V, M, Vb}(
            c,
            lv,
            uv,
            A,
            At,
            lc,
            uc,
            D1,
            D2,
            int_var,
            var_names,
            dataset,
            name,
            path
        )
    end
end

function MILP(qps::QPSData; kwargs...)
    return MILP(;
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

function Base.show(io::IO, milp::MILP{T, V, M}) where {T, V, M}
    return print(
        io, """
        MILP instance $(milp.name) 
        - types: values $T, vectors $V, matrices $M
        - variables: $(nbvar(milp)) ($(nbvar_cont(milp)) continuous, $(nbvar_int(milp)) integer)
        - constraints: $(nbcons(milp)) ($(nbcons_ineq(milp)) inequalities, $(nbcons_eq(milp)) equalities)
        - nonzeros: $(mynnz(milp.A))"""
    )
end

KernelAbstractions.get_backend(milp::MILP) = get_backend(milp.c)

"""
    nbvar(milp)

Return the number of variables in `milp`.
"""
nbvar(milp::MILP) = length(milp.c)

"""
    nbvar_int(milp)

Return the number of integer variables in `milp`.
"""
nbvar_int(milp::MILP) = sum(milp.int_var)

"""
    nbvar_cont(milp)

Return the number of integer variables in `milp`.
"""
nbvar_cont(milp::MILP) = nbvar(milp) - nbvar_int(milp)

"""
    nbcons(milp)

Return the number of constraints in `milp`, not including variable bounds or integrality requirements.
"""
nbcons(milp::MILP) = size(milp.A, 1)

"""
    nbcons_eq(milp)

Return the number of equality constraints in `milp`.
"""
nbcons_eq(milp::MILP) = mapreduce((l, u) -> (l == u), +, milp.lc, milp.uc)

"""
    nbcons_ineq(milp)

Return the number of inequality constraints in `milp`, not including variable bounds.
"""
nbcons_ineq(milp::MILP) = nbcons(milp) - nbcons_eq(milp)

function Base.isapprox(m1::MILP, m2::MILP; kwargs...)
    return (
        isapprox(m1.c, m2.c; kwargs...) &&
            isapprox(m1.lv, m2.lv; kwargs...) &&
            isapprox(m1.uv, m2.uv; kwargs...) &&
            isapprox(m1.A, m2.A; kwargs...) &&
            isapprox(m1.At, m2.At; kwargs...) &&
            isapprox(m1.lc, m2.lc; kwargs...) &&
            isapprox(m1.uc, m2.uc; kwargs...) &&
            isapprox(m1.D1, m2.D1; kwargs...) &&
            isapprox(m1.D2, m2.D2; kwargs...) &&
            m1.int_var == m2.int_var &&
            m1.var_names == m2.var_names &&
            m1.dataset == m2.dataset &&
            m1.name == m2.name &&
            m1.path == m2.path
    )
end
