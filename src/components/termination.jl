"""
    current_time()

Return the current host time in seconds, like `Base.time()`.

It is only ever called on the host: inside a Reactant-compiled solve, the clock is read by the
host callback behind [`check_time_limit!`](@ref) instead.
"""
current_time() = time()

"""
    HostClock

Wall-clock bookkeeping of a solve, kept on the host.

It is a separate object so that Reactant can leave it out of a compiled program (see the
Reactant extension): the times are `Float64`, which some accelerators cannot hold, and they are
only ever read and written by host code.

# Fields

$(TYPEDFIELDS)
"""
mutable struct HostClock
    "time at which the algorithm started, in seconds"
    starting_time::Float64
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64
end

"""
    CLOCK_REGISTRY

Every live [`HostClock`](@ref), by the id that [`register_clock!`](@ref) gave it, held weakly.

A compiled program cannot hold a host object, so a Reactant-compiled solve reaches the clock of
the state it runs on through its id, a traced number of [`ConvergenceStats`](@ref): a program
compiled on one state then updates the clock of whichever state it is called with.
"""
const CLOCK_REGISTRY = Dict{UInt32, WeakRef}()
const CLOCK_REGISTRY_LOCK = ReentrantLock()
const LAST_CLOCK_ID = Ref{UInt32}(0)

"""
    register_clock!(clock)

Give `clock` a new id in [`CLOCK_REGISTRY`](@ref) and return it. Entries of clocks which have
been garbage collected are swept every 256 registrations.
"""
function register_clock!(clock::HostClock)
    return @lock CLOCK_REGISTRY_LOCK begin
        id = LAST_CLOCK_ID[] += one(UInt32)
        if iszero(id % UInt32(256))
            filter!(entry -> entry.second.value !== nothing, CLOCK_REGISTRY)
        end
        CLOCK_REGISTRY[id] = WeakRef(clock)
        id
    end
end

"""
    update_registered_clock!(id, time_limit)

Update the elapsed time of the clock registered as `id`, and return whether it has reached
`time_limit`. A clock which is no longer registered never reaches it.
"""
function update_registered_clock!(id::Integer, time_limit::Float64)
    ref = @lock CLOCK_REGISTRY_LOCK get(CLOCK_REGISTRY, UInt32(id), nothing)
    clock = ref === nothing ? nothing : ref.value
    clock isa HostClock || return false
    clock.time_elapsed = current_time() - clock.starting_time
    return clock.time_elapsed >= time_limit
end

"""
    check_time_limit!(stats, time_limit)

Update the elapsed time in `stats` and return whether it has reached `time_limit`, which lives
in [`GenericParameters`](@ref).

The Reactant extension overlays this, so that inside a compiled solve the comparison happens on
the host, and only its `Bool` result enters the compiled program. With an infinite `time_limit`,
the compiled program does not read the clock at all.
"""
function check_time_limit!(stats, time_limit::Number)
    clock = stats.clock
    clock.time_elapsed = current_time() - clock.starting_time
    return clock.time_elapsed >= time_limit
end

"""
    TerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct TerminationParameters{T <: Number, I <: Number}
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::I
end

function Base.show(io::IO, params::TerminationParameters)
    (; termination_reltol, max_kkt_passes) = params
    return print(io, "TerminationParameters: termination_reltol=$termination_reltol, max_kkt_passes=$max_kkt_passes")
end


"""
    ConvergenceStats

The properties `starting_time` and `time_elapsed` are forwarded to the [`HostClock`](@ref) in
`clock`.

# Fields

$(TYPEDFIELDS)
"""
mutable struct ConvergenceStats{T <: BatchedNumber, K <: Number, I <: Number, S <: Number}
    "current KKT error"
    err::KKTErrors{T}
    "host-side starting time and elapsed time, left out of a Reactant-compiled program"
    clock::HostClock
    "id of `clock` in [`CLOCK_REGISTRY`](@ref), traced under Reactant so that a compiled solve reads the clock of the state it is called with"
    clock_id::K
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::I
    "integer code of the termination status (should be that of `MOI.OPTIMIZE_NOT_CALLED` until the algorithm actually terminates), read with [`termination_status`](@ref)"
    termination_status_code::S
    "history of KKT errors, indexed by number of KKT passes"
    error_history::Vector{Tuple{I, KKTErrors{T}}}
end

function ConvergenceStats(
        err::KKTErrors{T};
        starting_time = current_time(),
        time_elapsed = 0.0,
        kkt_passes::I = 0,
        termination_status_code::S = status_code(MOI.OPTIMIZE_NOT_CALLED),
        error_history = [(kkt_passes, copy(err))]
    ) where {T, I, S}
    clock = HostClock(starting_time, time_elapsed)
    clock_id = register_clock!(clock)
    return ConvergenceStats{T, UInt32, I, S}(
        err,
        clock,
        clock_id,
        kkt_passes,
        termination_status_code,
        error_history
    )
end

const CLOCK_PROPERTIES = (:starting_time, :time_elapsed)

function Base.getproperty(stats::ConvergenceStats, name::Symbol)
    if name in CLOCK_PROPERTIES
        return getfield(getfield(stats, :clock), name)
    else
        return getfield(stats, name)
    end
end

function Base.setproperty!(stats::ConvergenceStats, name::Symbol, x)
    if name in CLOCK_PROPERTIES
        return setproperty!(getfield(stats, :clock), name, x)
    else
        return setfield!(stats, name, convert(fieldtype(typeof(stats), name), x))
    end
end

Base.propertynames(::ConvergenceStats) = (fieldnames(ConvergenceStats)..., CLOCK_PROPERTIES...)

"""
    status_code(status)

Return the integer code of a `MOI.TerminationStatusCode`, as stored in the
`termination_status_code` field of [`ConvergenceStats`](@ref).

Unlike the enum itself, this code is an ordinary number, so Reactant can trace it: a compiled
solve can write its own status instead of freezing the trace-time one.
"""
status_code(status::MOI.TerminationStatusCode) = Int32(status)

"""
    termination_status(stats)

Return the termination status of `stats` as a `MOI.TerminationStatusCode`.

Decodes the `termination_status_code` field, which is stored as a plain number so that it
survives a Reactant-compiled solve. Only call this outside a compilation context: mid-trace the
code is a traced number with no value yet.
"""
function termination_status(stats::ConvergenceStats)
    return MOI.TerminationStatusCode(Int32(stats.termination_status_code))
end

function instance(stats::ConvergenceStats, i::Int)
    return ConvergenceStats(
        instance(stats.err, i);
        starting_time = stats.starting_time,
        time_elapsed = stats.time_elapsed,
        kkt_passes = stats.kkt_passes,
        termination_status_code = stats.termination_status_code,
        error_history = [(passes, instance(err, i)) for (passes, err) in stats.error_history],
    )
end

function Base.show(io::IO, stats::ConvergenceStats)
    (; err, time_elapsed, kkt_passes) = stats
    return print(
        io,
        """Convergence stats with termination status $(termination_status(stats)):
        - $err
        - time elapsed: $time_elapsed seconds
        - KKT passes: $kkt_passes""",
    )
end

"""
    set_termination_status!!(stats, dest, params, is_time_limit)

Decide how the algorithm terminates, using `dest` as scratch space for the relative errors.

Whether the time limit is reached comes from [`check_time_limit!`](@ref), since the limit lives
in [`GenericParameters`](@ref) and, inside a compiled solve, is checked on the host.

Set `stats.termination_status_code` and return whether the algorithm should stop.

The status is selected with nested `ifelse` calls rather than with branches, so that it survives
a Reactant-compiled solve: an assignment inside a `@trace if` only ever contributes its
trace-time value, whereas an `ifelse` over traced conditions is part of the compiled program.
The returned boolean is what the solve loops branch on, since a status code cannot drive traced
control flow.
"""
function set_termination_status!!(
        stats::ConvergenceStats,
        dest::BatchedNumber,
        params::TerminationParameters,
        is_time_limit
    )
    (; err, kkt_passes) = stats
    (; termination_reltol, max_kkt_passes) = params
    is_optimal = batched_all(<=(termination_reltol), relative!!(dest, err))
    is_iteration_limit = kkt_passes >= max_kkt_passes
    # nested `ifelse` instead of `if`/`elseif`: it is a plain (traceable) computation, and it
    # states the priority between simultaneous criteria in one place
    stats.termination_status_code = ifelse(
        is_optimal,
        status_code(MOI.OPTIMAL),
        ifelse(
            is_time_limit,
            status_code(MOI.TIME_LIMIT),
            ifelse(
                is_iteration_limit,
                status_code(MOI.ITERATION_LIMIT),
                status_code(MOI.OPTIMIZE_NOT_CALLED),
            ),
        ),
    )
    return is_optimal | is_time_limit | is_iteration_limit
end
