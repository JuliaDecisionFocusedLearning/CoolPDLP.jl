"""
    is_feasible(x, milp[; cons_tol=1e-6, int_tol=1e-5, verbose=true])

Check whether solution vector `x` is feasible for `milp`.

# Keyword arguments

- `cons_tol`: tolerance for constraint satisfaction
- `int_tol`: tolerance for integrality requirements
- `verbose`: whether to display warnings
"""
function is_feasible(
        x::AbstractVector{T}, milp::MILP; cons_tol = 1.0e-6, int_tol = 1.0e-5, verbose::Bool = true
    ) where {T}
    (; G, h, A, b, l, u, intvar) = milp
    eq_err = maximum(abs, A * x - b; init = typemin(T))
    ineq_err = maximum(h - G * x; init = typemin(T))
    bounds_err = max(maximum(x - u), maximum(l - x))
    xint = x[intvar]
    int_err = maximum(abs, xint .- round.(Int, xint))
    if eq_err > cons_tol
        verbose && @warn "Equality constraints not satisfied" eq_err cons_tol
        return false
    elseif ineq_err > cons_tol
        verbose && @warn "Inequality constraints not satisfied" ineq_err cons_tol
        return false
    elseif bounds_err > cons_tol
        verbose && @warn "Variable bounds not satisfied" bounds_err cons_tol
        return false
    elseif int_err > int_tol
        verbose && @warn "Integrality not satisfied" int_err cons_tol
        return false
    else
        return true
    end
end

"""
    objective_value(x, milp)

Compute the value of the linear objective of `milp` at solution vector `x`.
"""
function objective_value(x::AbstractVector, milp::MILP)
    return dot(x, milp.c)
end
