"""
    is_feasible(x, milp[; cons_tol=1e-6, int_tol=1e-5, verbose=true])

Check whether solution vector `x` is feasible for `milp`.

# Keyword arguments

- `cons_tol`: tolerance for constraint satisfaction
- `int_tol`: tolerance for integrality requirements
- `verbose`: whether to display warnings
"""
function is_feasible(
        x, milp::MILP;
        cons_tol = 1.0e-6, int_tol = 1.0e-5, verbose::Bool = true
    )
    (; lv, uv, A, lc, uc, int_var) = milp
    bounds_err = max(maximum(x - uv), maximum(lv - x))
    cons_err = max(maximum(A * x - uc), maximum(lc - A * x))
    xint = x[int_var]
    int_err = maximum(abs, xint .- round.(Int, xint))
    if bounds_err > cons_tol
        verbose && @warn "Variable bounds not satisfied" bounds_err cons_tol
        return false
    elseif cons_err > cons_tol
        verbose && @warn "Constraints not satisfied" cons_err cons_tol
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
objective_value(x, milp::MILP) = dot(x, milp.c)

"""
    PrimalDualSolution

# Fields

$(TYPEDFIELDS)
"""
mutable struct PrimalDualSolution{T <: Number, V <: DenseVector{T}}
    "primal solution"
    const x::V
    "dual solution"
    const y::V
end

Base.eltype(::PrimalDualSolution{T}) where {T} = T

function Base.copy(z::PrimalDualSolution)
    return PrimalDualSolution(
        copy(z.x),
        copy(z.y),
    )
end

function Base.zero(z::PrimalDualSolution{T}) where {T}
    return PrimalDualSolution(
        zero(z.x),
        zero(z.y),
    )
end

function zero!(z::PrimalDualSolution{T}) where {T}
    zero!(z.x)
    zero!(z.y)
    return nothing
end

function Base.copy!(z1::PrimalDualSolution, z2::PrimalDualSolution)
    copy!(z1.x, z2.x)
    copy!(z1.y, z2.y)
    return z1
end

function LinearAlgebra.axpby!(
        a::T, x::PrimalDualSolution{T, V}, b::T, y::PrimalDualSolution{T, V},
    ) where {T, V}
    axpby!(a, x.x, b, y.x)
    axpby!(a, x.y, b, y.y)
    return y
end

function Base.isapprox(sol1::PrimalDualSolution{T, V}, sol2::PrimalDualSolution{T, V}; kwargs...) where {T, V}
    return isapprox(sol1.x, sol2.x; kwargs...) && isapprox(sol1.y, sol2.y; kwargs...)
end
