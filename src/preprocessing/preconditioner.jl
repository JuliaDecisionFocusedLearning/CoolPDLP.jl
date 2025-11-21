struct Preconditioner
    D1::Diagonal{Float64, Vector{Float64}}
    D2::Diagonal{Float64, Vector{Float64}}
end

function Base.:*(pb::Preconditioner, pa::Preconditioner)
    return Preconditioner(pb.D1 * pa.D1, pa.D2 * pb.D2)
end

function apply(preconditioner::Preconditioner, K, Kᵀ)
    (; D1, D2) = preconditioner
    K̃ = D1 * K * D2
    K̃ᵀ = D2 * Kᵀ * D1
    return K̃, K̃ᵀ
end

function identity_preconditioner(K::SparseMatrixCSC)
    d1 = ones(size(K, 1))
    d2 = ones(size(K, 2))
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function diagonal_norm_preconditioner(
        K::SparseMatrixCSC, Kᵀ::SparseMatrixCSC; p_row::Number, p_col::Number
    )
    col_norms = map(j -> column_norm(K, j, p_col), axes(K, 2))
    row_norms = map(i -> column_norm(Kᵀ, i, p_row), axes(K, 1))
    d1 = map(rn -> iszero(rn) ? 1.0 : inv(sqrt(rn)), row_norms)
    d2 = map(cn -> iszero(cn) ? 1.0 : inv(sqrt(cn)), col_norms)
    return Preconditioner(Diagonal(d1), Diagonal(d2))
end

function chambolle_pock_preconditioner(K, Kᵀ; α::Number)
    return diagonal_norm_preconditioner(K, Kᵀ; p_row = 2 - α, p_col = α)
end

function ruiz_preconditioner(K, Kᵀ; iterations::Integer)
    p_acc = identity_preconditioner(K)
    for _ in 1:iterations
        p = diagonal_norm_preconditioner(K, Kᵀ; p_col = Inf, p_row = Inf)
        K, Kᵀ = apply(p, K, Kᵀ)
        p_acc = p * p_acc
    end
    return p_acc
end

function preconditioned_solution(preconditioner::Preconditioner, x::Vector, y::Vector)
    (; D1, D2) = preconditioner
    return D2 \ x, D1 * y
end

function unpreconditioned_solution(preconditioner::Preconditioner, x::Vector, y::Vector)
    (; D1, D2) = preconditioner
    return D2 * x, D1 \ y
end
