"""
    relax(milp)

Return a new `MILP` identical to `milp` but without integrality requirements.
"""
relax(milp::MILP) = @set milp.int_var = zero(milp.int_var)

"""
    set_eltype(T, milp)

Change the element type of floating-point containers inside `milp` to `T`.
"""
set_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = map(T, A)

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
    set_indtype(T, milp)

Change the element type of integer containers inside `milp` to `T`.
"""
set_indtype(::Type{T}, A::AbstractArray{<:Integer}) where {T} = map(T, A)

function set_indtype(::Type{T}, A::SparseMatrixCSC) where {T}
    return SparseMatrixCSC(
        A.m,
        A.n,
        set_indtype(T, A.colptr),
        set_indtype(T, A.rowval),
        A.nzval
    )
end

function set_indtype(::Type{T}, milp::MILP) where {T}
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c,
        lv,
        uv,
        A = set_indtype(T, A),
        At = set_indtype(T, At),
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
    single_precision(milp)

Convert all integers in `milp` to `Int32` and all floating-point numbers to `Float32`.
"""
single_precision(x) = set_eltype(Float32, set_indtype(Int32, x))

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
