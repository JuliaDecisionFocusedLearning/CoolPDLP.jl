"""
    TerminationReason

Enum type listing possible reasons for algorithm termination:

- `CONVERGENCE`
- `TIME`
- `ITERATIONS`
- `STILL_RUNNING`
"""
@enum TerminationReason CONVERGENCE TIME ITERATIONS STILL_RUNNING

"""
    AbstractState

Algorithm state supertype.

!!! warning
    Work in progress.

# Required fields

- `time_elapsed`
- `kkt_passes`
- `err`
- `termination_reason`
"""
abstract type AbstractState{T <: Number, V <: AbstractVector{T}} end

function Base.show(io::IO, state::AbstractState)
    (; err, time_elapsed, kkt_passes, termination_reason) = state
    return print(
        io,
        """$(nameof(typeof(state))) with termination reason $termination_reason:
        - relative KKT error: $(relative(err))
        - time elapsed: $(round(time_elapsed; digits = 3)) seconds 
        - KKT passes: $kkt_passes""",
    )
end

"""
    AbstractParameters

Algorithm parameter supertype.

!!! warning
    Work in progress.

# Required fields

- `termination_reltol`
- `time_limit`
- `max_kkt_passes`
- `record_error_history`
"""
abstract type AbstractParameters{T <: Number} end

"""
    KKTErrors

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct KKTErrors{T <: Number}
    primal::T
    dual::T
    gap::T
    primal_scale::T
    dual_scale::T
    gap_scale::T
    rel_max::T
    weighted_agg::T
end

function Base.show(io::IO, err::KKTErrors)
    return print(
        io,
        """KKT errors:
        - primal $(err.primal) (scale $(err.primal_scale))
        - dual $(err.dual) (scale $(err.dual_scale))
        - gap $(err.gap) (scale $(err.gap_scale))""",
    )
end

function KKTErrors(::Type{T}) where {T}
    return KKTErrors(
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
    )
end

relative(err::KKTErrors) = err.rel_max
absolute(err::KKTErrors) = err.weighted_agg

function termination_check!(state::AbstractState, params)
    (; err) = state
    (; termination_reltol, time_limit, max_kkt_passes) = params

    if relative(err) <= termination_reltol
        state.termination_reason = CONVERGENCE
        return true
    elseif state.time_elapsed >= time_limit
        state.termination_reason = TIME
        return true
    elseif state.kkt_passes >= max_kkt_passes
        state.termination_reason = ITERATIONS
        return true
    else
        state.termination_reason = STILL_RUNNING
        return false
    end
end
