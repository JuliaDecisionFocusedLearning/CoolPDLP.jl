"""
    set_eltype(T, milp)

Change the element type of floating-point containers inside `milp` to `T`.
"""
set_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = map(T, A)

function set_eltype(::Type{T}, sol::PrimalDualSolution) where {T}
    return PrimalDualSolution(set_eltype(T, sol.x), set_eltype(T, sol.y))
end

function set_eltype(::Type{T}, milp::MILP) where {T}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = set_eltype(T, c),
        lv = set_eltype(T, lv),
        uv = set_eltype(T, uv),
        A = set_eltype(T, A),
        At = set_eltype(T, At),
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
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c,
        lv,
        uv,
        A = set_indtype(Ti, A),
        At = set_indtype(Ti, At),
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

"""
    set_matrix_type(::Type{M}, milp)

Convert the sparse matrices inside `milp` using constructor `M`.
"""
function set_matrix_type(::Type{M}, milp::MILP) where {M}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c,
        lv,
        uv,
        A = M(A),
        At = M(At),
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

function Adapt.adapt_structure(to, sol::PrimalDualSolution)
    return PrimalDualSolution(adapt(to, sol.x), adapt(to, sol.y))
end

function Adapt.adapt_structure(to, milp::MILP)
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = adapt(to, c),
        lv = adapt(to, lv),
        uv = adapt(to, uv),
        A = adapt(to, A),
        At = adapt(to, At),
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
