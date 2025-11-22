"""
    is_feasible(x, milp[; cons_tol=1e-6, int_tol=1e-5, verbose=true])

Check whether solution vector `x` is feasible for `milp`.

# Keyword arguments

- `cons_tol`: tolerance for constraint satisfaction
- `int_tol`: tolerance for integrality requirements
- `verbose`: whether to display warnings
"""
function is_feasible(
        x::AbstractVector{T}, milp::MILP;
        cons_tol = 1.0e-6, int_tol = 1.0e-5, verbose::Bool = true
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
objective_value(x::AbstractVector, milp::MILP) = dot(x, milp.c)


"""
    read_sol(path::String, milp::MILP)

Read a solution stored in a `.sol` file following the MIPLIB specification, return a vector of floating-point numbers.
"""
function read_sol(path::String, milp::MILP)
    T = Float64
    x = fill(convert(T, NaN), nbvar(milp))
    open(path, "r") do f
        obj = NaN
        i = 1
        for line in eachline(f)
            if isempty(line)
                continue
            elseif startswith(line, "=obj=")
                obj = parse(T, split(line)[end])
            else
                v = parse(T, split(line)[end])  # TODO: handle more generic separators?
                x[i] = v
                i += 1
            end
        end
        @assert objective_value(x, milp) ≈ obj
    end
    return x
end

"""
    write_sol(path::String, x::AbstractVector, milp::MILP)

Write a solution to a `.sol` file following the MIPLIB specification.
"""
function write_sol(path::String, x::AbstractVector{<:Number}, milp::MILP)
    @assert endswith(path, ".sol")
    x = float.(x)
    open(path, "w") do f
        print(f, "=obj= ")
        print(f, objective_value(x, milp))
        ret = "\n"
        space = " "
        for i in eachindex(x)
            print(f, ret, milp.varname[i], space, x[i])
        end
    end
    return
end

"""
    PrimalDualSolution

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PrimalDualSolution{T <: Number, V <: AbstractVector{T}}
    const x::V
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
