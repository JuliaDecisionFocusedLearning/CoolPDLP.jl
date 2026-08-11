function random_milp_and_sol(rng::Random.AbstractRNG, m::Int, n::Int, p::Float64)
    c = rand(rng, n)
    A = sprandn(rng, m, n, p)
    luv = randn(rng, n)
    luc = randn(rng, m)
    lv = map(luv) do z
        r = rand(rng)
        if r < 0.25
            return z
        elseif r < 0.5
            return -Inf
        else
            return z - rand(rng)
        end
    end
    uv = map(luv) do z
        r = rand(rng)
        if r < 0.25
            return z
        elseif r < 0.5
            return +Inf
        else
            return z + rand(rng)
        end
    end
    lc = map(luc) do z
        r = rand(rng)
        if r < 0.25
            return z
        elseif r < 0.5
            return -Inf
        else
            return z - rand(rng)
        end
    end
    uc = map(luc) do z
        r = rand(rng)
        if r < 0.25
            return z
        elseif r < 0.5
            return +Inf
        else
            return z + rand(rng)
        end
    end
    int_var = rand(rng, Bool, length(c))
    x = clamp.(randn(rng, n), lv, uv)
    y = proj_multiplier.(randn(rng, m), lc, uc)
    return MILP(; c, lv, uv, A, lc, uc, int_var), PrimalDualSolution(x, y)
end

function random_milp_and_sol(m::Int, n::Int, p::Float64)
    return random_milp_and_sol(Random.default_rng(), m, n, p)
end
