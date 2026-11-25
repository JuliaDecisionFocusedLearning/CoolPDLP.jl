"""
    set_eltype(T, milp)

Change the element type of floating-point containers inside `milp` to `T`.
"""
set_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = map(T, A)
set_eltype(::Type{T}, Z::ZeroMatrix) where {T} = ZeroMatrix{T}(Z.n)
set_eltype(::Type{T}, S::BatchScaledMatrix) where {T} = BatchScaledMatrix(set_eltype(T, S.Q), set_eltype(T, S.scale))

function set_eltype(::Type{T}, sol::PrimalDualSolution) where {T}
    return PrimalDualSolution(set_eltype(T, sol.x), set_eltype(T, sol.y))
end

function set_eltype(::Type{T}, milp::MILP) where {T}
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = set_eltype(T, c),
        c0 = T(c0),
        lv = set_eltype(T, lv),
        uv = set_eltype(T, uv),
        A = set_eltype(T, A),
        At = set_eltype(T, At),
        Q = set_eltype(T, Q),
        lc = set_eltype(T, lc),
        uc = set_eltype(T, uc),
        D1 = set_eltype(T, D1),
        D2 = set_eltype(T, D2),
        bound_scale,
        obj_scale,
        int_var,
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end

"""
    set_indtype(Ti, milp)

Change the element type of integer containers inside `milp` to `Ti`.
"""
set_indtype(::Type{Ti}, A::AbstractArray{<:Integer}) where {Ti} = map(Ti, A)
set_indtype(::Type, Z::ZeroMatrix) = Z
set_indtype(::Type{Ti}, S::BatchScaledMatrix) where {Ti} = BatchScaledMatrix(set_indtype(Ti, S.Q), S.scale)

function set_indtype(::Type{Ti}, A::SparseMatrixCSC) where {Ti}
    return SparseMatrixCSC(
        A.m,
        A.n,
        set_indtype(Ti, A.colptr),
        set_indtype(Ti, A.rowval),
        A.nzval
    )
end

function set_indtype(::Type{Ti}, milp::MILP) where {Ti}
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c,
        c0,
        lv,
        uv,
        A = set_indtype(Ti, A),
        At = set_indtype(Ti, At),
        Q = set_indtype(Ti, Q),
        lc,
        uc,
        D1,
        D2,
        bound_scale,
        obj_scale,
        int_var,
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end

"""
    convert_matrix(M, Q)

Convert the objective matrix `Q` with constructor `M`, unless it is a [`ZeroMatrix`](@ref).

A sparse `Q` stays sparse with dense matrices (`M = Matrix`), in the [`GPUSparseMatrixCSR`](@ref)
format, which also compiles under Reactant: a dense `Q` would always count as quadratic (see
[`isquadratic`](@ref)), so the empty `Q` of a linear program read from a file would cost a dense
product at every step.
"""
convert_matrix(::Type{M}, Q::AbstractMatrix) where {M} = M(Q)
convert_matrix(::Type{<:Matrix}, Q::SparseMatrixCSC) = GPUSparseMatrixCSR(Q)
convert_matrix(::Type, Z::ZeroMatrix) = Z
convert_matrix(::Type{M}, S::BatchScaledMatrix) where {M} = BatchScaledMatrix(convert_matrix(M, S.Q), S.scale)

"""
    set_matrix_type(::Type{M}, milp)

Convert the sparse matrices inside `milp` using constructor `M`.
"""
function set_matrix_type(::Type{M}, milp::MILP) where {M}
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    A_M = M(A)
    At_M = M(At)
    Q_M = convert_matrix(M, Q)
    backend = common_backend(A_M, At_M)

    return MILP(;
        c = adapt(backend, c),
        c0,
        lv = adapt(backend, lv),
        uv = adapt(backend, uv),
        A = A_M,
        At = At_M,
        Q = Q_M,
        lc = adapt(backend, lc),
        uc = adapt(backend, uc),
        D1 = adapt(backend, D1),
        D2 = adapt(backend, D2),
        bound_scale = adapt(backend, bound_scale),
        obj_scale = adapt(backend, obj_scale),
        int_var = adapt(backend, int_var),
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end

function Adapt.adapt_structure(to, sol::PrimalDualSolution)
    return PrimalDualSolution(adapt(to, sol.x), adapt(to, sol.y))
end

function Adapt.adapt_structure(to, milp::MILP)
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = adapt(to, c),
        c0,
        lv = adapt(to, lv),
        uv = adapt(to, uv),
        A = adapt(to, A),
        At = adapt(to, At),
        Q = adapt(to, Q),
        lc = adapt(to, lc),
        uc = adapt(to, uc),
        D1 = adapt(to, D1),
        D2 = adapt(to, D2),
        bound_scale = adapt(to, bound_scale),
        obj_scale = adapt(to, obj_scale),
        int_var = adapt(to, int_var),
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end

"""
    relax(milp)

Return the continuous relaxation of `milp`, i.e. the same problem with its integrality
restrictions dropped.

`solve` only ever tackles that relaxation (see [`solve`](@ref)), so a tool which does apply
integer-specific transformations — a presolver, say — must be handed the relaxation rather than
`milp` itself, on pain of working on a problem nobody is solving.
"""
function relax(milp::MILP)
    (;
        c, c0, lv, uv, A, At, Q, lc, uc, D1, D2, bound_scale, obj_scale,
        int_var, var_names, con_names, dataset, name, path,
    ) = milp
    return MILP(;
        c,
        c0,
        lv,
        uv,
        A,
        At,
        Q,
        lc,
        uc,
        D1,
        D2,
        bound_scale,
        obj_scale,
        int_var = zero!(similar(int_var)),
        var_names,
        con_names,
        dataset,
        name,
        path
    )
end
