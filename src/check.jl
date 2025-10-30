function is_feasible(
        x::AbstractVector{<:Number}, milp::MILP{T};
        cons_tol = 1.0e-6, int_tol = 1.0e-5, verbose::Bool = true
    ) where {T}
    (; G, h, A, b, l, u, intvar) = milp
    eq_err = maximum(abs, A * x - b)
    ineq_err = maximum(h - G * x)
    bounds_err = max(maximum(x - u), maximum(l - x))
    xint = x[intvar]
    int_err = maximum(abs, xint .- round.(Int, xint))
    if eq_err > cons_tol
        verbose && @warn "Equality constraints not satisfied"
        return false
    elseif ineq_err > cons_tol
        verbose && @warn "Inequality constraints not satisfied"
        return false
    elseif bounds_err > cons_tol
        verbose && @warn "Variable bounds not satisfied"
        return false
    elseif bin_err > int_tol || int_err > int_tol
        verbose && @warn "Integrality not satisfied"
        return false
    else
        return true
    end
end

function objective_value(x::AbstractVector{<:Number}, milp::MILP)
    return dot(x, milp.c)
end
