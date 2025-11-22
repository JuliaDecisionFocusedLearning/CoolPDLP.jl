"""
    relax(milp)

Return a new `MILP` identical to `milp` but without integrality requirements.
"""
relax(milp::MILP) = @set milp.int_var = zero(milp.int_var)

"""
    precondition(milp, D1, D2)
"""
function precondition(milp::MILP, D1::Diagonal, D2::Diagonal)
    (;
        c, lv, uv, A, At, lc, uc,
        int_var, var_names, dataset, name, path,
    ) = milp
    return MILP(;
        c = D2 * c,
        lv = D2 * lv,
        uv = D2 * uv,
        A = D1 * A * D2,
        At = D2 * At * D1,
        lc = D1 * lc,
        uc = D1 * uc,
        D1 = D1 * milp.D1,
        D2 = milp.D2 * D2,
        int_var,
        var_names,
        dataset,
        name,
        path
    )
end

"""
    sort_rows_columns(milp)
"""
function sort_rows_columns(milp::MILP)
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp

    perm_var = increasing_column_order(A)
    perm_cons = increasing_column_order(At)

    return MILP(;
        c = c[perm_var],
        lv = lv[perm_var],
        uv = uv[perm_var],
        A = permute_rows(permute_columns(A, perm_var), perm_cons),
        At = permute_rows(permute_columns(At, perm_cons), perm_var),
        lc = lc[perm_cons],
        uc = uc[perm_cons],
        D1 = Diagonal(D1.diag[perm_cons]),
        D2 = Diagonal(D2.diag[perm_var]),
        int_var = int_var[perm_var],
        var_names = var_names[perm_var],
        dataset,
        name,
        path
    )
end

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
function set_indtype(::Type{M}, milp::MILP) where {M}
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
