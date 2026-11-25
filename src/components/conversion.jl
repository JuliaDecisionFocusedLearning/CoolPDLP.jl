"""
    ConversionParameters{T,Ti,M,B,Tm}

# Type parameters

- `T`: floating point type to convert values to
- `Ti`: integer type to convert indices to
- `M`: matrix constructor to use on the constraints
- `B`: backend type
- `Tm`: floating point type to store the matrices `A` and `Aᵀ` in, or `Nothing` for `T`

# Fields

$(TYPEDFIELDS)
"""
struct ConversionParameters{
        T <: Union{Number, Nothing},
        Ti <: Union{Integer, Nothing},
        M <: Union{AbstractMatrix, Nothing},
        B <: Union{Backend, Nothing},
        Tm <: Union{Number, Nothing},
    }
    "CPU or GPU backend used for computations, or `nothing` to avoid any backend conversion"
    backend::B

    function ConversionParameters(
            ::Type{T},
            ::Type{Ti},
            ::Type{M},
            ::Type{Tm} = Nothing;
            backend::B,
        ) where {T, Ti, M, B, Tm}
        return new{T, Ti, M, B, Tm}(
            backend,
        )
    end
end

ConversionParameters() = ConversionParameters(Nothing, Nothing, Nothing; backend = nothing)

function Base.show(io::IO, params::ConversionParameters{T, Ti, M, B, Tm}) where {T, Ti, M, B, Tm}
    (; backend) = params
    matrix = Tm <: Nothing ? "" : ", matrix_eltype=$Tm"
    return print(io, "ConversionParameters: types=($T, $Ti, $M)$matrix, backend=$backend")
end

function perform_conversion(
        milp::MILP,
        params::ConversionParameters{T, Ti, M, B, Tm},
    ) where {T, Ti, M, B, Tm}
    (; backend) = params
    if !(T == Nothing)
        milp = set_eltype(T, milp)
    end
    if !(Ti == Nothing)
        milp = set_indtype(Ti, milp)
    end
    # after `set_eltype`, so that the vectors keep `T`, and before `set_matrix_type`, so that
    # the matrix formats are built directly in `Tm`
    if !(Tm == Nothing)
        milp = set_matrix_eltype(Tm, milp)
    end
    if !(M == Nothing)
        milp = set_matrix_type(M, milp)
    end
    if isnothing(backend)
        return milp
    else
        return adapt(backend, milp)
    end
end

function perform_conversion(
        sol::PrimalDualSolution,
        params::ConversionParameters{T},
    ) where {T}
    (; backend) = params
    if !(T == Nothing)
        sol = set_eltype(T, sol)
    end
    if isnothing(backend)
        return sol
    else
        return adapt(backend, sol)
    end
end
