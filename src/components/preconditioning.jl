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
    return print(io, "PreconditioningParameters: chambolle_pock_alpha=$chambolle_pock_alpha, ruiz_iter=$ruiz_iter")
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

function Base.:*(prec_out::Preconditioner, prec_in::Preconditioner)
    return Preconditioner(prec_out.D1 * prec_in.D1, prec_in.D2 * prec_out.D2)
end

Base.inv(prec::Preconditioner) = Preconditioner(inv(prec.D1), inv(prec.D2))

function precondition_matrix(
        A::AbstractMatrix, At::AbstractMatrix, prec::Preconditioner
    )
    (; D1, D2) = prec
    A_p = D1 * A * D2
    At_p = D2 * At * D1
    return A_p, At_p
end

function precondition_variables(
        x::AbstractVector, y::AbstractVector, prec::Preconditioner
    )
    (; D1, D2) = prec
    x_p = D2 \ x
    # original problem: c - At * y = r
    # preconditioned problem: c̃ - Ãt * ỹ = r̃
    # c̃ = D2 * c, Ãt = D2 * At * D1
    # D2 * (c - At * D1 * ỹ) = r̃
    # makes me want to take ỹ = D1 \ y and r̃ = D2 * r
    y_p = D1 \ y
    return (x_p, y_p)
end

function precondition_problem(
        milp::MILP, prec::Preconditioner
    )
    (;
        c, lv, uv, A, At, lc, uc,
        int_var, var_names, dataset, name, path,
    ) = milp
    (; D1, D2) = prec
    c_p = D2 * c
    lv_p, uv_p = D2 \ lv, D2 \ uv
    A_p, At_p = precondition_matrix(A, At, prec)
    lc_p, uc_p = D1 * lc, D1 * uc
    new_prec = prec * Preconditioner(milp.D1, milp.D2)
    milp_p = MILP(;
        c = c_p,
        lv = lv_p,
        uv = uv_p,
        A = A_p,
        At = At_p,
        lc = lc_p,
        uc = uc_p,
        D1 = new_prec.D1,
        D2 = new_prec.D2,
        int_var,
        var_names,
        dataset,
        name,
        path
    )
    return milp_p
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
    prec = identity_preconditioner(A, At)
    for _ in 1:iterations
        prec_next = diagonal_norm_preconditioner(A, At; p_col = Inf, p_row = Inf)
        A, At = precondition_matrix(A, At, prec_next)
        prec = prec_next * prec
    end
    return prec
end

function pdlp_preconditioner(milp::MILP, params::PreconditioningParameters)
    (; A, At) = milp
    (; chambolle_pock_alpha, ruiz_iter) = params
    prec_r = ruiz_preconditioner(A, At; iterations = ruiz_iter)
    A_r, At_r = precondition_matrix(A, At, prec_r)
    prec_cp = chambolle_pock_preconditioner(A_r, At_r; alpha = chambolle_pock_alpha)
    prec = prec_r * prec_cp
    return prec
end
