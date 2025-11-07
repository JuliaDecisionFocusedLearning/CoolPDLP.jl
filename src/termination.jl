function individual_kkt_errors!(
        state::AbstractState{Tv, V},
        sad::SaddlePointProblem{Tv, V},
        x::V,
        y::V,
    ) where {Tv, V}
    (; x_scratch1, x_scratch2, x_scratch3, y_scratch) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad

    qᵀy = dot(q, y)
    cᵀx = dot(c, x)

    # λ = proj_Λ(c - Kᵀ * y)  from cuPDLP-C paper
    λ = x_scratch1
    λ .= c
    mul!(λ, Kᵀ, y, -Tv(1), Tv(1))
    proj_Λ!(λ, l, u)

    λ⁺, l_noinf = x_scratch2, x_scratch3
    λ⁺ .= positive_part.(λ)
    l_noinf .= ifelse.(iszero.(λ⁺), zero(Tv), l)
    lᵀλ⁺ = dot(l_noinf, λ⁺)

    λ⁻, u_noinf = x_scratch2, x_scratch3
    λ⁻ .= negative_part.(λ)
    u_noinf .= ifelse.(iszero.(λ⁻), zero(Tv), u)
    uᵀλ⁻ = dot(u_noinf, λ⁻)

    # Kxᵀ = (Gxᵀ, Axᵀ), qᵀ = (hᵀ, bᵀ)
    err_primal_scratch = y_scratch
    err_primal_scratch .= q
    mul!(err_primal_scratch, K, x, -Tv(1), Tv(1))
    err_primal_scratch .= ifelse.(
        ineq_cons, positive_part.(err_primal_scratch), err_primal_scratch
    )

    # err_primal = sqrt(sqnorm(b - Ax) + sqnorm((h - Gx)⁺)
    err_primal = norm(err_primal_scratch)
    err_primal_denominator = one(Tv) + norm(q)

    # err_dual = norm(c - Kᵀ * y - λ)
    err_dual_scratch = x_scratch1
    err_dual_scratch .= c .- λ
    mul!(err_dual_scratch, Kᵀ, y, -Tv(1), Tv(1))

    err_dual = norm(err_dual_scratch)
    err_dual_denominator = one(Tv) + norm(c)

    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    err_gap_denominator = one(Tv) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    return (;
        err_primal,
        err_dual,
        err_gap,
        err_primal_denominator,
        err_dual_denominator,
        err_gap_denominator,
    )
end

function max_relative_kkt_error!(
        state::AbstractState{Tv, V},
        sad::SaddlePointProblem{Tv, V},
        x::V,
        y::V,
    ) where {Tv, V}
    (;
        err_primal, err_dual, err_gap,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors!(state, sad, x, y)

    relative_error_primal = err_primal / err_primal_denominator
    relative_error_dual = err_dual / err_dual_denominator
    relative_error_gap = err_gap / err_gap_denominator

    return max(relative_error_primal, relative_error_dual, relative_error_gap)
end

function aggregated_absolute_kkt_error!(
        state::AbstractState{Tv, V},
        sad::SaddlePointProblem{Tv, V},
        x::V,
        y::V,
        ω::Number
    ) where {Tv, V}
    (; err_primal, err_dual, err_gap) = individual_kkt_errors!(state, sad, x, y)
    err_agg = sqrt(ω^2 * err_primal^2 + inv(ω^2) * err_dual^2 + err_gap^2)
    return err_agg
end

function termination_check!(
        state::AbstractState,
        sad::SaddlePointProblem,
        params::AbstractParameters,
    )
    (; starting_time) = state
    (; termination_reltol, time_limit, max_kkt_passes, record_error_history) = params
    state.time_elapsed = time() - starting_time
    state.relative_error = max_relative_kkt_error!(state, sad, state.x, state.y)
    if record_error_history
        push!(state.relative_error_history, (state.kkt_passes, state.relative_error))
    end

    if state.relative_error <= termination_reltol
        state.termination_reason = CONVERGENCE
        return true
    elseif state.time_elapsed >= time_limit
        state.termination_reason = TIME
        return true
    elseif state.kkt_passes >= max_kkt_passes
        state.termination_reason = ITERATIONS
        return true
    else
        state.termination_reason = STILL_RUNNING
        return false
    end
end
