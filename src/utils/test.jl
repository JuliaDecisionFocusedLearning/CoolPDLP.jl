function random_milp_and_sol(m::Int, n::Int, p::Float64)
    c = rand(n)
    A = sprandn(m, n, p)
    luv = randn(n)
    luc = randn(m)
    lv = map(luv) do z
        r = rand()
        if r < 0.25
            return z
        elseif r < 0.5
            return -Inf
        else
            return z - rand()
        end
    end
    uv = map(luv) do z
        r = rand()
        if r < 0.25
            return z
        elseif r < 0.5
            return +Inf
        else
            return z + rand()
        end
    end
    lc = map(luc) do z
        r = rand()
        if r < 0.25
            return z
        elseif r < 0.5
            return -Inf
        else
            return z - rand()
        end
    end
    uc = map(luc) do z
        r = rand()
        if r < 0.25
            return z
        elseif r < 0.5
            return +Inf
        else
            return z + rand()
        end
    end
    int_var = rand(Bool, length(c))
    x = proj_box.(randn(n), lv, uv)
    y = proj_multiplier.(randn(m), lc, uc)
    return MILP(; c, lv, uv, A, lc, uc, int_var), PrimalDualSolution(x, y)
end
