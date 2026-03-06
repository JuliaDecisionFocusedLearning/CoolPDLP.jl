"""
    set_eltype(T, milp)

Change the element type of floating-point containers inside `milp` to `T`.
"""
set_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = map(T, A)
set_eltype(::Type, ::Nothing) = nothing

function set_eltype(::Type{T}, sol::PrimalDualSolution) where {T}
    return PrimalDualSolution(set_eltype(T, sol.x), set_eltype(T, sol.y))
end

function set_eltype(::Type{T}, milp::AbstractProgram) where {T}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return rebuild(
        milp;
        c = set_eltype(T, c),
        lv = set_eltype(T, lv),
        uv = set_eltype(T, uv),
        A = set_eltype(T, A),
        At = set_eltype(T, At),
        Q = set_eltype(T, get_Q(milp)),
        lc = set_eltype(T, lc),
        uc = set_eltype(T, uc),
        D1 = set_eltype(T, D1),
        D2 = set_eltype(T, D2),
        int_var,
        var_names,
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
set_indtype(::Type, ::Nothing) = nothing

function set_indtype(::Type{Ti}, A::SparseMatrixCSC) where {Ti}
    return SparseMatrixCSC(
        A.m,
        A.n,
        set_indtype(Ti, A.colptr),
        set_indtype(Ti, A.rowval),
        A.nzval
    )
end

function set_indtype(::Type{Ti}, milp::AbstractProgram) where {Ti}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return rebuild(
        milp;
        c,
        lv,
        uv,
        A = set_indtype(Ti, A),
        At = set_indtype(Ti, At),
        Q = set_indtype(Ti, get_Q(milp)),
        lc, uc, D1, D2,
        int_var,
        var_names,
        dataset,
        name,
        path
    )
end

"""
    set_matrix_type(::Type{M}, milp)

Convert the sparse matrices inside `milp` using constructor `M`.
"""
function set_matrix_type(::Type{M}, milp::AbstractProgram) where {M}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    Q = get_Q(milp)
    A_M = set_matrix_type(M, A)
    At_M = set_matrix_type(M, At)
    Q_M = set_matrix_type(M, Q)
    backend = common_backend(A_M, At_M, Q_M)

    return rebuild(
        milp;
        c = adapt(backend, c),
        lv = adapt(backend, lv),
        uv = adapt(backend, uv),
        A = A_M,
        At = At_M,
        Q = Q_M,
        lc = adapt(backend, lc),
        uc = adapt(backend, uc),
        D1 = adapt(backend, D1),
        D2 = adapt(backend, D2),
        int_var = adapt(backend, int_var),
        var_names, dataset, name, path
    )
end

set_matrix_type(::Type{M}, mat) where {M} = M(mat)
set_matrix_type(::Type, ::Nothing) = nothing

function Adapt.adapt_structure(to, sol::PrimalDualSolution)
    return PrimalDualSolution(adapt(to, sol.x), adapt(to, sol.y))
end

function Adapt.adapt_structure(to, milp::AbstractProgram)
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    Q = get_Q(milp)
    return rebuild(
        milp;
        c = adapt(to, c),
        lv = adapt(to, lv),
        uv = adapt(to, uv),
        A = adapt(to, A),
        At = adapt(to, At),
        Q = adapt(to, Q),
        lc = adapt(to, lc),
        uc = adapt(to, uc),
        D1 = adapt(to, D1),
        D2 = adapt(to, D2),
        int_var = adapt(to, int_var),
        var_names,
        dataset,
        name,
        path
    )
end
