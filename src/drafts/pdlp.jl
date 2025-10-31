@kwdef struct PDLPParameters{T <: Number}
    β_sufficient::T = 0.2
    β_necessary::T = 0.8
    β_artificial::T = 0.36
    ϵ_zero::T = 1.0e-8
    ϵ_termination::T = 1.0e-4
    θ::T = 0.5
end

@kwdef mutable struct PDLPState{T}
    η::T
    ω::T
    converged::Bool = false
    kkt_passes::Int = 0
end

function custom_sqnorm(z::PrimalDualVariable{T}, ω::T) where {T}
    return sqrt(ω * sqnorm(z.x) + inv(ω) * sqnorm(z.y))
end

function pdlp(
        problem::SaddlePointProblem{T},
        param::PDLPParameters{T},
        z_init::PrimalDualVariable{T} = default_init(problem)
    ) where {T}
    (; c, q, K) = problem
    n = 1
    k = 1
    η̂ = inv(opnorm(K, Inf))
    ω = initialize_primal_weight(param, c, q)
    z⁰ = copy(z_init)
    for _ in 1:1  # outer loop on n
        yield()
        t = 1
        η_sum = zero(T)
        z = copy(z⁰)
        z̄ = zero(z)
        zᶜ = copy(z)
        for _ in 1:1000  # inner loop on t
            yield()
            z, η, η̂ = adaptive_step_pdhg(problem, z, ω, η̂, k)
            z̄ = inv(η_sum + η) * (η_sum * z̄ + η * z)
            η_sum += η
            zᶜ_next = get_restart_candidate(problem, z, z̄, ω)
            t += 1
            k += 1
            if restart_criterion(problem, param, zᶜ_next, zᶜ, z⁰; ω, k, t)
                break
            elseif termination_criterion(problem, param, z)
                return z
            else
                zᶜ = zᶜ_next
            end
        end
        ω = primal_weight_update(param, zᶜ, z⁰, ω)
        z⁰ = zᶜ
        if termination_criterion(problem, param, z⁰)
            return z⁰
        else
            n += 1
        end
    end
    return z⁰
end

function adaptive_step_pdhg(
        problem::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
        ω::T,
        η̂::T,
        k::Integer
    ) where {T}
    (; c, q, K, Kᵀ) = problem
    (; x, y) = z
    η = η̂
    for _ in 1:100 # TODO: infinite
        xp = proj_X(problem, x - (η / ω) * (c - Kᵀ * y))
        yp = proj_Y(problem, y + (η * ω) * (q - K * (2 * xp - x)))
        xdiff = xp - x
        ydiff = yp - y
        zdiff = PrimalDualVariable(xdiff, ydiff)
        η̄ = custom_sqnorm(zdiff, ω) / abs(2 * dot(ydiff, K, xdiff))
        ηp = min(
            (1 - (k + 1)^T(-0.3)) * η̄,
            (1 + (k + 1)^T(-0.6)) * η
        )
        if η <= η̄
            return PrimalDualVariable(xp, yp), η, ηp
        else
            η = ηp
        end
    end
    return nothing
end

function initialize_primal_weight(
        param::PDLPParameters{T},
        c::AbstractVector{T},
        q::AbstractVector{T}
    ) where {T}
    (; ϵ_zero) = param
    cn = norm(c)
    qn = norm(q)
    if cn > ϵ_zero && qn > ϵ_zero
        return cn / qn
    else
        return one(T)
    end
end

function primal_weight_update(
        param::PDLPParameters{T},
        z⁰_next::PrimalDualVariable{T},
        z⁰_last::PrimalDualVariable{T},
        ω_last::T,
    ) where {T}
    (; ϵ_zero, θ) = param
    x⁰_next, y⁰_next = z⁰_next.x, z⁰_next.y
    x⁰_last, y⁰_last = z⁰_last.x, z⁰_last.y

    Δx = norm(x⁰_next - x⁰_last)
    Δy = norm(y⁰_next - y⁰_last)

    if Δx > ϵ_zero && Δy > ϵ_zero
        return exp(θ * log(Δy / Δx) + (1 - θ) * log(ω_last))
    else
        return ω_last
    end
end

function kkt_error(
        problem::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
        ω::T
    ) where {T}
    (; err_primal_squared, err_dual_squared, err_gap_abs) = individual_kkt_errors(problem, z)
    return sqrt(
        abs2(ω) * err_primal_squared +
            inv(abs2(ω)) * err_dual_squared +
            abs2(err_gap_abs)
    )
end

function kkt_relative_error(
        problem::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
    ) where {T}
    (;
        err_primal_squared, err_dual_squared, err_gap_abs,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors(problem, z)
    rel_error_primal = sqrt(err_primal_squared) / err_primal_denominator
    rel_error_dual = sqrt(err_dual_squared) / err_dual_denominator
    rel_error_gap = err_gap_abs / err_gap_denominator
    return max(rel_error_primal, rel_error_dual, rel_error_gap)
end

function get_restart_candidate(
        problem::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
        z̄::PrimalDualVariable{T},
        ω::T
    ) where {T}
    err_z = kkt_error(problem, z, ω)
    err_z̄ = kkt_error(problem, z̄, ω)
    if err_z < err_z̄
        return z
    else
        return z̄
    end
end

function restart_criterion(
        problem::SaddlePointProblem{T},
        param::PDLPParameters{T},
        zᶜ_next::PrimalDualVariable{T},
        zᶜ::PrimalDualVariable{T},
        z⁰::PrimalDualVariable{T};
        ω::T,
        k::Integer,
        t::Integer
    ) where {T}
    return false
    (; β_sufficient, β_necessary, β_artificial) = param
    err_zᶜ_next = kkt_error(problem, zᶜ_next, ω)
    err_zᶜ = kkt_error(problem, zᶜ, ω)
    err_z⁰ = kkt_error(problem, z⁰, ω)
    if err_zᶜ_next <= β_sufficient * err_z⁰
        return true  # sufficient decay in KKT error
    elseif err_zᶜ_next <= β_necessary * err_z⁰ && err_zᶜ_next > err_zᶜ
        return true  # necessary decay + no local progress in KKT error
    elseif t > β_artificial * k
        return true  # long inner loop
    else
        return false
    end
end
