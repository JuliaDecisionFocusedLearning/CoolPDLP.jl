@kwdef struct PDLPParameters{T <: Number}
    β_sufficient::T = 0.2
    β_necessary::T = 0.8
    β_artificial::T = 0.36
    ϵ_zero::T = 1.0e-8
    ϵ_termination::T = 1.0e-4
    θ::T = 0.5
end

struct PrimalDualVariable{T <: Number, V <: AbstractVector{T}}
    x::V
    y::V
end

Base.copy(z::PrimalDualVariable) = PrimalDualVariable(copy(z.x), copy(z.y))
Base.zero(z::PrimalDualVariable) = PrimalDualVariable(zero(z.x), zero(z.y))

Base.:*(α::T, z::PrimalDualVariable{T}) where {T <: Number} = PrimalDualVariable(α * z.x, α * z.y)

function Base.:+(z1::PrimalDualVariable{T}, z2::PrimalDualVariable{T}) where {T <: Number}
    return PrimalDualVariable(z1.x + z2.x, z1.y + z2.y)
end

function pdlp(
        sad::SaddlePointProblem{T},
        par::PDLPParameters{T},
        z_init::PrimalDualVariable{T} = PrimalDualVariable(zero(sad.c), zero(sad.q))
    ) where {T <: Number}
    (; c, q, K) = sad
    n = 1
    k = 1
    η̂ = inv(opnorm(K, Inf))
    ω = initialize_primal_weight(par, c, q)
    z⁰ = copy(z_init)
    for _ in 1:100  # outer loop on n
        yield()
        t = 1
        η_sum = zero(T)
        z = copy(z⁰)
        z̄ = zero(z)
        zᶜ = copy(z)
        for _ in 1:100  # inner loop on t
            yield()
            z, η, η̂ = adaptive_step_pdhg(sad, par, z, ω, η̂, k)
            z̄ = inv(η_sum + η) * (η_sum * z̄ + η * z)
            η_sum += η
            zᶜ_next = get_restart_candidate(sad, z, z̄, ω)
            t += 1
            k += 1
            if restart_criterion(sad, par, zᶜ_next, zᶜ, z⁰; ω, k, t)
                break
            elseif termination_criterion(sad, par, z)
                return z
            else
                zᶜ = zᶜ_next
            end
        end
        ω = primal_weight_update(par, zᶜ, z⁰, ω)
        z⁰ = zᶜ
        if termination_criterion(sad, par, z⁰)
            return z⁰
        else
            n += 1
        end
    end
    return z⁰
end

function proj_X(sad::SaddlePointProblem{T}, x::AbstractVector{T}) where {T <: Number}
    (; l, u) = sad
    return min.(max.(x, l), u)
end

function proj_Y(sad::SaddlePointProblem{T}, y::AbstractVector{T}) where {T <: Number}
    (; m₁, m₂) = sad
    return vcat(positive_part(y[1:m₁]), y[(m₁ + 1):(m₁ + m₂)])
end

@inline positive_part(a::T) where {T <: Number} = max(a, zero(T))
@inline negative_part(a::T) where {T <: Number} = -min(a, zero(T))

positive_part(a::AbstractVector{T}) where {T <: Number} = positive_part.(a)
negative_part(a::AbstractVector{T}) where {T <: Number} = negative_part.(a)

function proj_λ(lᵢ::T, uᵢ::T, λᵢ::T) where {T <: Number}
    if lᵢ == typemin(T) && uᵢ == typemax(T)
        return zero(T)  # project on {0}
    elseif lᵢ == typemin(T)
        return -negative_part(λᵢ)  # project on ℝ⁻
    elseif uᵢ == typemax(T)
        return positive_part(λᵢ)  # project on ℝ⁺
    else
        return λᵢ  # project on ℝ
    end
end

function proj_Λ(sad::SaddlePointProblem{T}, λ::AbstractVector{T}) where {T <: Number}
    (; l, u) = sad
    return map(proj_λ, l, u, λ)
end

sqnorm(v::AbstractVector{<:Number}) = dot(v, v)

function custom_sqnorm(z::PrimalDualVariable{T}, ω::T) where {T <: Number}
    return sqrt(ω * sqnorm(z.x) + inv(ω) * sqnorm(z.y))
end

function adaptive_step_pdhg(
        sad::SaddlePointProblem{T},
        par::PDLPParameters{T},
        z::PrimalDualVariable{T},
        ω::T,
        η̂::T,
        k::Integer
    ) where {T <: Number}
    (; c, q, K, Kᵀ) = sad
    (; x, y) = z
    η = η̂
    for it in 1:100 # TODO: infinite
        xp = proj_X(sad, x - (η / ω) * (c - Kᵀ * y))
        yp = proj_Y(sad, y + (η * ω) * (q - K * (2 * xp - x)))
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
        par::PDLPParameters{T},
        c::AbstractVector{T},
        q::AbstractVector{T}
    ) where {T <: Number}
    (; ϵ_zero) = par
    cn = norm(c)
    qn = norm(q)
    if cn > ϵ_zero && qn > ϵ_zero
        return cn / qn
    else
        return one(T)
    end
end

function primal_weight_update(
        par::PDLPParameters{T},
        z⁰_next::PrimalDualVariable{T},
        z⁰_last::PrimalDualVariable{T},
        ω_last::T,
    ) where {T <: Number}
    (; ϵ_zero, θ) = par
    x⁰_next, y⁰_next = z⁰_next.x, z⁰_last.y
    x⁰_last, y⁰_last = z⁰_last.x, z⁰_last.y

    Δx = norm(x⁰_next - x⁰_last)
    Δy = norm(y⁰_next - y⁰_last)

    if Δx > ϵ_zero && Δy > ϵ_zero
        return exp(θ * log(Δy / Δx) + (1 - θ) * ω_last)
    else
        return ω_last
    end
end

function individual_kkt_errors(
        sad::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
    ) where {T <: Number}
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad
    (; x, y) = z

    Kx = K * x
    Gx, h = Kx[1:m₁], q[1:m₁]
    Ax, b = Kx[(m₁ + 1):(m₁ + m₂)], q[(m₁ + 1):(m₁ + m₂)]

    λ = proj_Λ(sad, c - Kᵀ * y)  # from cuPDLP-C paper
    λ⁺ = positive_part(λ)
    λ⁻ = negative_part(λ)

    qᵀy = dot(q, y)
    lᵀλ⁺ = dot(l, λ⁺)
    uᵀλ⁻ = dot(u, λ⁻)
    cᵀx = dot(c, x)

    err_primal_squared = sqnorm(Ax - b) + sqnorm(positive_part.(h - Gx))
    err_dual_squared = sqnorm(c - Kᵀ * y - λ)
    err_gap_abs = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)

    err_primal_denominator = 1 + norm(q)
    err_dual_denominator = 1 + norm(c)
    err_gap_denominator = 1 + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    return (;
        err_primal_squared, err_dual_squared, err_gap_abs,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    )
end

function kkt_error(
        sad::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
        ω::T
    ) where {T <: Number}
    (; err_primal_squared, err_dual_squared, err_gap_abs) = individual_kkt_errors(sad, z)
    return sqrt(
        abs2(ω) * err_primal_squared +
            inv(abs2(ω)) * err_dual_squared +
            abs2(err_gap_abs)
    )
end

function kkt_relative_error(
        sad::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
    ) where {T <: Number}
    (;
        err_primal_squared, err_dual_squared, err_gap_abs,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors(sad, z)
    rel_error_primal = sqrt(err_primal_squared) / err_primal_denominator
    rel_error_dual = sqrt(err_dual_squared) / err_dual_denominator
    rel_error_gap = err_gap_abs / err_gap_denominator
    return max(rel_error_primal, rel_error_dual, rel_error_gap)
end

function get_restart_candidate(
        sad::SaddlePointProblem{T},
        z::PrimalDualVariable{T},
        z̄::PrimalDualVariable{T},
        ω::T
    ) where {T <: Number}
    err_z = kkt_error(sad, z, ω)
    err_z̄ = kkt_error(sad, z̄, ω)
    if err_z < err_z̄
        return z
    else
        return z̄
    end
end

function restart_criterion(
        sad::SaddlePointProblem{T},
        par::PDLPParameters{T},
        zᶜ_next::PrimalDualVariable{T},
        zᶜ::PrimalDualVariable{T},
        z⁰::PrimalDualVariable{T};
        ω::T,
        k::Integer,
        t::Integer
    ) where {T <: Number}
    return false
    (; β_sufficient, β_necessary, β_artificial) = par
    err_zᶜ_next = kkt_error(sad, zᶜ_next, ω)
    err_zᶜ = kkt_error(sad, zᶜ, ω)
    err_z⁰ = kkt_error(sad, z⁰, ω)
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

function termination_criterion(
        sad::SaddlePointProblem{T},
        par::PDLPParameters{T},
        z::PrimalDualVariable{T},
    ) where {T <: Number}
    (; ϵ_termination) = par
    rel_err = kkt_relative_error(sad, z)
    return rel_err <= ϵ_termination
end
