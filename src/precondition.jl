"""
    precondition(sad::SaddlePointProblem, D1, D2)

Apply preconditioning with matrices `(D1, D2)` to `sad`.
"""
function precondition(sad::SaddlePointProblem{T}, D1::AbstractMatrix, D2::AbstractMatrix) where {T}
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    K̃ = D1 * K * D2
    K̃ᵀ = D2 * Kᵀ * D1
    c̃ = D2 * c
    q̃ = D1 * q
    l̃ = D2 \ l
    ũ = D2 \ u
    l̃_noinf = max.(nextfloat(typemin(T)), l̃)
    ũ_noinf = min.(prevfloat(typemax(T)), ũ)
    D̃1 = D1 * sad.D1
    D̃2 = sad.D2 * D2
    return SaddlePointProblem(;
        c = c̃, q = q̃,
        K = K̃, Kᵀ = K̃ᵀ,
        l = l̃, u = ũ,
        l_noinf = l̃_noinf, u_noinf = ũ_noinf,
        ineq_cons,
        D1 = D̃1, D2 = D̃2
    )
end

function diagonal_norm_preconditioner(
        sad::SaddlePointProblem{T}, p_row::Number, p_col::Number
    ) where {T}
    (; c, q, K, Kᵀ) = sad
    d1 = similar(q)
    d2 = similar(c)
    for j in axes(K, 1)
        if isempty(column_view(Kᵀ, j))
            d1[j] = one(T)
        else
            d1[j] = sqrt(norm(column_view(Kᵀ, j), T(p_row))) |> inv  # missing in PDLP paper
        end
    end
    for i in axes(K, 2)
        if isempty(column_view(K, i))
            d2[i] = one(T)
        else
            d2[i] = sqrt(norm(column_view(K, i), T(p_col))) |> inv  # missing in PDLP paper
        end
    end
    return Diagonal(d1), Diagonal(d2)
end

function precondition_chambolle_pock(sad::SaddlePointProblem{T}; α::Number) where {T}
    D1, D2 = diagonal_norm_preconditioner(sad, T(2 - α), T(α))
    return precondition(sad, D1, D2)
end

function precondition_ruiz(sad::SaddlePointProblem{T}; iterations::Integer) where {T}
    sad_iter = sad
    for _ in 1:iterations
        D1, D2 = diagonal_norm_preconditioner(sad_iter, Inf, Inf)
        sad_iter = precondition(sad_iter, D1, D2)
    end
    return sad_iter
end

function preconditioned_solution(
        sad::SaddlePointProblem,
        x::AbstractVector, y::AbstractVector,
    )
    (; D1, D2) = sad
    return D2 \ x, D1 * y
end

function unpreconditioned_solution(
        sad::SaddlePointProblem,
        x::AbstractVector, y::AbstractVector,
    )
    (; D1, D2) = sad
    return D2 * x, D1 \ y
end
