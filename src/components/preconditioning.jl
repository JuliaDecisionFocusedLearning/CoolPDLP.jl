"""
    PreconditioningParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PreconditioningParameters{T}
    "norm parameter in the Chambolle-pock preconditioner"
    chambolle_pock_alpha::T
    "iteration parameter in the Ruiz preconditioner"
    ruiz_iter::Int
end

function Base.show(io::IO, params::PreconditioningParameters)
    (; chambolle_pock_alpha, ruiz_iter) = params
    return print(io, "Preconditioning: chambolle_pock_alpha=$chambolle_pock_alpha, ruiz_iter=$ruiz_iter")
end

# Preconditioning effect

"""
    Preconditioner

# Fields

$(TYPEDFIELDS)
"""
struct Preconditioner{T <: Number, V <: AbstractVector{T}}
    "left preconditioner"
    D1::Diagonal{T, V}
    "right preconditioner"
    D2::Diagonal{T, V}
end

function Base.:*(pout::Preconditioner, pin::Preconditioner)
    return Preconditioner(pout.D1 * pin.D1, pin.D2 * pout.D2)
end

function precondition_matrix(
        A::AbstractMatrix, At::AbstractMatrix, p::Preconditioner
    )
    (; D1, D2) = p
    A_precond = D1 * A * D2
    At_precond = D2 * At * D1
    return A_precond, At_precond
end

function unprecondition_matrix(
        A_precond::AbstractMatrix, At_precond::AbstractMatrix, p::Preconditioner
    )
    (; D1, D2) = p
    A = inv(D1) * A_precond * inv(D2)
    At = inv(D2) * At_precond * inv(D1)
    return A, At
end

precondition_primal(x::AbstractVector, p::Preconditioner) = p.D2 \ x
unprecondition_primal(x::AbstractVector, p::Preconditioner) = p.D2 * x

precondition_dual(y::AbstractVector, p::Preconditioner) = p.D1 * y
unprecondition_dual(y::AbstractVector, p::Preconditioner) = p.D1 \ y

function precondition_variables(
        x::AbstractVector, y::AbstractVector, p::Preconditioner
    )
    return precondition_primal(x, p), precondition_dual(y, p)
end

function unprecondition_variables(
        x::AbstractVector, y::AbstractVector, p::Preconditioner
    )
    return unprecondition_primal(x, p), unprecondition_dual(y, p)
end

function precondition(
        milp::MILP, x::AbstractVector, y::AbstractVector, p::Preconditioner
    )
    (;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path,
    ) = milp
    c_precond = unprecondition_primal(c, p)
    lv_precond = precondition_primal(lv, p)
    uv_precond = precondition_primal(uv, p)
    A_precond, At_precond = precondition_matrix(A, At, p)
    lc_precond = precondition_dual(lc, p)
    uc_precond = precondition_dual(uc, p)
    new_p = p * Preconditioner(D1, D2)
    milp_precond = MILP(;
        c = c_precond,
        lv = lv_precond,
        uv = uv_precond,
        A = A_precond,
        At = At_precond,
        lc = lc_precond,
        uc = uc_precond,
        D1 = new_p.D1,
        D2 = new_p.D2,
        int_var,
        var_names,
        dataset,
        name,
        path
    )
    x_precond, y_precond = precondition_variables(x, y, p)
    return milp_precond, x_precond, y_precond
end

# Preconditioner construction

function identity_preconditioner(
        A::AbstractMatrix, At::AbstractMatrix
    )
    d1 = ones(size(A, 1))
    d2 = ones(size(A, 2))
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function diagonal_norm_preconditioner(
        A::SparseMatrixCSC{T}, At::SparseMatrixCSC{T};
        p_row::Number, p_col::Number
    ) where {T}
    col_norms = map(j -> column_norm(A, j, p_col), axes(A, 2))
    row_norms = map(i -> column_norm(At, i, p_row), axes(A, 1))
    d1 = map(rn -> iszero(rn) ? one(T) : inv(sqrt(rn)), row_norms)
    d2 = map(cn -> iszero(cn) ? one(T) : inv(sqrt(cn)), col_norms)
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function chambolle_pock_preconditioner(A, At; alpha::Number)
    return diagonal_norm_preconditioner(A, At; p_row = 2 - alpha, p_col = alpha)
end

function ruiz_preconditioner(A, At; iterations::Integer)
    p = identity_preconditioner(A, At)
    for _ in 1:iterations
        p_next = diagonal_norm_preconditioner(A, At; p_col = Inf, p_row = Inf)
        A, At = precondition_matrix(A, At, p_next)
        p = p_next * p
    end
    return p
end

function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At) = milp
    (; chambolle_pock_alpha, ruiz_iter) = params
    p_r = ruiz_preconditioner(A, At; iterations = ruiz_iter)
    A_r, At_r = precondition_matrix(A, At, p_r)
    p_cp = chambolle_pock_preconditioner(A_r, At_r; alpha = chambolle_pock_alpha)
    p = p_r * p_cp
    return p
end
