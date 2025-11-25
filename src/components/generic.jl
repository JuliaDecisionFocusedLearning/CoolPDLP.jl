"""
    GenericParameters{T,Ti,M}

# Type parameters

- `T`: floating point type to convert to
- `Ti`: integer index type to convert to
- `M`: sparse matrix type to convert to

# Fields

$(TYPEDFIELDS)
"""
struct GenericParameters{
        T <: Number, Ti <: Integer, M <: AbstractMatrix, B <: Backend,
    }
    "CPU or GPU backend used for computations"
    backend::B
    "tolerance in absolute comparisons to zero"
    zero_tol::T
    "frequency of restart or termination checks"
    check_every::Int
    "whether or not to record error evolution"
    record_error_history::Bool

    function GenericParameters(
            ::Type{T},
            ::Type{Ti},
            ::Type{M},
            backend::B;
            zero_tol,
            check_every,
            record_error_history
        ) where {T, Ti, M, B}
        return new{T, Ti, M, B}(
            backend,
            zero_tol,
            check_every,
            record_error_history
        )
    end
end

function Base.show(io::IO, params::GenericParameters)
    (; backend, zero_tol, check_every, record_error_history) = params
    return print(io, "Generic: backend=$backend, zero_tol=$zero_tol, check_every=$check_every, record_error_history=$record_error_history")
end

function to_device(
        milp::MILP,
        params::GenericParameters{T, Ti, M},
    ) where {T, Ti, M}
    (; backend) = params
    milp_righttypes = set_matrix_type(M, set_indtype(Ti, set_eltype(T, milp)))
    milp_adapted = adapt(backend, milp_righttypes)
    return milp_adapted
end

function to_device(
        x::AbstractVector,
        params::GenericParameters{T, Ti, M},
    ) where {T, Ti, M}
    (; backend) = params
    x_righttype = set_eltype(T, x)
    x_adapted = adapt(backend, x_righttype)
    return x_adapted
end
