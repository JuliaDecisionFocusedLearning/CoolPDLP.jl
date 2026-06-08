"""
    ConversionParameters{T,Ti,M}

# Type parameters

- `T`: floating point type to convert values to
- `Ti`: integer type to convert indices to
- `M`: matrix constructor to use on the constraints

# Fields

$(TYPEDFIELDS)
"""
struct ConversionParameters{
        T <: Number, Ti <: Integer, M <: AbstractMatrix, B <: Backend,
    }
    "CPU or GPU backend used for computations"
    backend::B

    function ConversionParameters(
            ::Type{T},
            ::Type{Ti},
            ::Type{M};
            backend::B
        ) where {T, Ti, M, B}
        return new{T, Ti, M, B}(
            backend,
        )
    end
end

function Base.show(io::IO, params::ConversionParameters{T, Ti, M}) where {T, Ti, M}
    (; backend) = params
    return print(io, "ConversionParameters: types=($T, $Ti, $M), backend=$backend")
end

function perform_conversion(
        milp::MILP,
        params::ConversionParameters{T, Ti, M},
    ) where {T, Ti, M}
    (; backend) = params
    milp_righttypes = set_matrix_type(M, set_indtype(Ti, set_eltype(T, milp)))
    milp_adapted = adapt(backend, milp_righttypes)
    return milp_adapted
end

function perform_conversion(
        sol::PrimalDualSolution,
        params::ConversionParameters{T},
    ) where {T}
    (; backend) = params
    sol_righttypes = set_eltype(T, sol)
    sol_adapted = adapt(backend, sol_righttypes)
    return sol_adapted
end
