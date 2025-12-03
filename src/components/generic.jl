"""
    GenericParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct GenericParameters
    "whether to show a progress bar"
    show_progress::Bool
    "frequency of restart or termination checks"
    check_every::Int
    "whether or not to record error evolution"
    record_error_history::Bool
end

function Base.show(io::IO, params::GenericParameters)
    (; show_progress, check_every, record_error_history) = params
    return print(io, "GenericParameters: show_progress=$show_progress, check_every=$check_every, record_error_history=$record_error_history")
end
