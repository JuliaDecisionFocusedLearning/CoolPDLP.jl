"""
    precondition(sad::SaddlePointProblem, D1, D2)

Apply preconditioning with matrices `(D1, D2)` to `sad`.
"""
function precondition(sad::SaddlePointProblem, D1::AbstractMatrix, D2::AbstractMatrix)
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    K̃ = D1 * K * D2
    K̃ᵀ = D2 * Kᵀ * D1
    c̃ = D2 * c
    q̃ = D1 * q
    l̃ = D2 \ l
    ũ = D2 \ u
    D̃1 = D1 * sad.D1
    D̃2 = sad.D2 * D2
    return SaddlePointProblem(;
        c = c̃, q = q̃,
        K = K̃, Kᵀ = K̃ᵀ,
        l = l̃, u = ũ,
        ineq_cons,
        D1 = D̃1, D2 = D̃2
    )
end

function diagonal_norm_preconditioner(
        sad::SaddlePointProblem{Tv}, p_row::Number, p_col::Number
    ) where {Tv}
    (; c, q, K, Kᵀ) = sad
    d1 = similar(q)
    d2 = similar(c)
    for j in axes(K, 1)
        d1[j] = sqrt(norm(column_view(Kᵀ, j), Tv(p_row))) |> inv  # missing in PDLP paper
    end
    for i in axes(K, 2)
        d2[i] = sqrt(norm(column_view(K, i), Tv(p_col))) |> inv  # missing in PDLP paper
    end
    return Diagonal(d1), Diagonal(d2)
end

function precondition_chambolle_pock(sad::SaddlePointProblem{Tv}; α::Number) where {Tv}
    D1, D2 = diagonal_norm_preconditioner(sad, Tv(2 - α), Tv(α))
    return precondition(sad, D1, D2)
end

function precondition_ruiz(sad::SaddlePointProblem{Tv}; iterations::Integer) where {Tv}
    sad_iter = sad
    for _ in 1:iterations
        D1, D2 = diagonal_norm_preconditioner(sad_iter, Inf, Inf)
        sad_iter = precondition(sad_iter, D1, D2)
    end
    return sad_iter
end

"""
    precondition_pdlp(sad::SaddlePointProblem)

Apply the default PDLP preconditioning to `sad`: a few iterations of Ruiz scaling, followed by Chambolle-Pock scaling.
"""
function precondition_pdlp(
        sad::SaddlePointProblem; ruiz_iterations = 10, chambolle_pock_alpha = 1
    )
    sad_ruiz = precondition_ruiz(sad; iterations = ruiz_iterations)
    sad_cp = precondition_chambolle_pock(sad_ruiz; α = chambolle_pock_alpha)
    return sad_cp
end
