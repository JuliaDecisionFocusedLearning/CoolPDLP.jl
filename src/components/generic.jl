"""
    GenericParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct GenericParameters
    "frequency of restart or termination checks"
    check_every::Int
    "whether or not to record error evolution"
    record_error_history::Bool
end

function Base.show(io::IO, params::GenericParameters)
    (; check_every, record_error_history) = params
    return print(io, "GenericParameters: check_every=$check_every, record_error_history=$record_error_history")
end
