"""
    GenericParameters

Parameters that shape the solve loop itself rather than the values it computes. Under
Reactant, they are compile-time constants: see the Reactant section of the README.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct GenericParameters{I <: Number, B, F <: Number}
    "whether to show a progress bar"
    show_progress::B
    "frequency of restart or termination checks"
    check_every::I
    "whether or not to record error evolution"
    record_error_history::B
    "time limit in seconds (`Inf` for none); a Reactant-compiled solve only reads the clock when it is finite"
    time_limit::F
    "norm in which the termination check measures the primal and dual residuals, `:L2` or `:Linf`; a Reactant-compiled solve only contains the selected reduction"
    termination_norm::Symbol = :L2
end

function Base.show(io::IO, params::GenericParameters)
    (; show_progress, check_every, record_error_history, time_limit, termination_norm) = params
    return print(io, "GenericParameters: show_progress=$show_progress, check_every=$check_every, record_error_history=$record_error_history, time_limit=$time_limit, termination_norm=$termination_norm")
end
