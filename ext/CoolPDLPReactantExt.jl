module CoolPDLPReactantExt

using CoolPDLP: CoolPDLP
using LinearAlgebra: LinearAlgebra
using Reactant: Reactant, @reactant_overlay

# A sparse matrix's shape and pattern are structure, and so are the generic parameters: with a
# traced `check_every`, the inner loop of `solve!` has a dynamic trip count, and XLA reads its
# condition back to the host after every KKT pass; whether `time_limit` is finite decides
# whether the compiled program reads the clock at all, through a host callback (see
# `CoolPDLP.check_time_limit!`). So is the backend in the conversion parameters, whose `Bool`
# fields (e.g. in `CUDABackend` or `CPU`) cannot hold traced numbers. Trace them without number
# tracking: their values are baked into the compiled program.
const Structural = Union{
    CoolPDLP.GPUSparseMatrix, CoolPDLP.GenericParameters, CoolPDLP.ConversionParameters,
}

# The clock is host data through and through: tracing leaves it as it is, so that no `Float64`
# enters a compiled program (some accelerators, like Metal, cannot hold one). A compiled solve
# reaches it through the traced `clock_id` of the stats instead.
function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:CoolPDLP.HostClock}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime),
    )
    return T
end

function Reactant.make_tracer(
        seen, @nospecialize(prev::CoolPDLP.HostClock), @nospecialize(path), mode; kwargs...,
    )
    mode == Reactant.TracedToTypes && (push!(path, typeof(prev)); return nothing)
    return prev
end

function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:Structural}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime),
    )
    return @invoke Reactant.traced_type_inner(
        T::Type, seen::Any, mode::Reactant.TraceMode, Union{}::Type, ndevices::Any, runtime::Any
    )
end

function Reactant.make_tracer(
        seen,
        @nospecialize(prev::Structural),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        kwargs...,
    )
    return Reactant.make_tracer_unknown(
        seen, prev, path, mode; track_numbers = Union{}, kwargs...
    )
end

"""
    TimeLimitCheck(time_limit)

Host callback behind the overlay of [`CoolPDLP.check_time_limit!`](@ref): updates the elapsed
time of the clock whose id it gets as input (see [`CoolPDLP.CLOCK_REGISTRY`](@ref)), and
writes whether it has reached `time_limit` into its single `UInt8` output.

The clock is looked up by id at each call rather than captured while tracing, so a compiled
`solve!` called on another state than the one it was compiled with updates the clock of that
other state. `time_limit` is a constant of the compiled program, like the other generic
parameters.

`Reactant.Ops.julia_callback` hands the callback its output buffers first and its inputs
afterwards. A `()`-shaped output arrives dereferenced, with nothing to write into, hence the
`(1,)` shape, while the scalar id arrives as a value. Uses `fill!` because CUDA.jl refuses
`out[1] = ...` on a device buffer.
"""
struct TimeLimitCheck
    time_limit::Float64
end

function (check::TimeLimitCheck)(out::AbstractVector{UInt8}, id)
    reached = CoolPDLP.update_registered_clock!(scalar_id(id), check.time_limit)
    fill!(out, reached)
    return nothing
end

scalar_id(id::Integer) = UInt32(id)
scalar_id(id::AbstractArray{<:Integer}) = UInt32(only(Array(id)))

"""
    closure_cfunctions_supported()

Whether this Julia can build the closure `@cfunction` that `Reactant.Ops.julia_callback` uses.

Mirrors the check in Julia's codegen (`src/codegen.cpp`, unchanged from 1.10 to 1.13): closure
cfunctions need LLVM trampolines, and are refused on AArch64, 32-bit ARM and PPC64 targets,
where tracing a callback fails with "cfunction: closures are not supported on this platform".
"""
closure_cfunctions_supported() =
    !(Sys.ARCH === :aarch64 || Sys.ARCH === :powerpc64le || startswith(String(Sys.ARCH), "arm"))

"""
    host_callbacks_supported()

Whether `Reactant.Ops.julia_callback` can be serviced on the backend currently in use.

`Reactant.Ops._wrap_buffers` hands the callback its buffers directly on the host, goes through
`CUDA.jl` on the CUDA backend, and raises on every other one. A callback that raises is caught
inside Reactant's trampoline, which logs it and reports failure *on every call* rather than
stopping the run, so an unusable callback surfaces as a hang rather than as an error. Better
not to emit one at all.
"""
function host_callbacks_supported()
    closure_cfunctions_supported() || return false
    platform = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    platform == "cpu" && return true
    platform == "cuda" && return Reactant.is_extension_loaded(Val(:CUDA))
    return false
end

function warn_no_time_limit()
    @warn """
    This platform or Reactant backend cannot run a host callback, so a solve compiled with \
    `@compile` cannot read the clock: `time_limit` will not be enforced and `time_elapsed` \
    stays at its initial value. Call `solve!` without `@compile` for a host-driven solve, which \
    enforces it on every backend.""" maxlog = 1
    return nothing
end

"""
    traced_time_limit_check(stats, time_limit)

Body of the overlay of [`CoolPDLP.check_time_limit!`](@ref).
"""
function traced_time_limit_check(stats::CoolPDLP.ConvergenceStats, time_limit::Number)
    # both branches return a traced `Bool`, so that `termination_check!` stays type-stable.
    # `time_limit` is a host value here (the generic parameters are structure), so an infinite
    # one needs no clock at all: that is how the host-driven `solve!` below compiles its
    # launches, enforcing the limit itself
    isinf(time_limit) && return Reactant.Ops.constant(false)::Reactant.TracedRNumber{Bool}
    host_callbacks_supported() ||
        return (warn_no_time_limit(); Reactant.Ops.constant(false)::Reactant.TracedRNumber{Bool})
    out = Reactant.Ops.julia_callback(
        TimeLimitCheck(Float64(time_limit)), ((UInt8, (1,)),), stats.clock_id;
        has_side_effect = true,
    )
    # a reduction, since a traced array cannot be indexed, and `any` rather than `sum`, which
    # would widen the `UInt8` to a `UInt64` on the device
    return any(!iszero, out)::Reactant.TracedRNumber{Bool}
end

"""
    ConcreteRVecOrMat{T}

Vectors and matrices that hold device data outside of a compilation, as opposed to traced arrays.
"""
const ConcreteRVecOrMat{T} = Union{
    Reactant.ConcretePJRTArray{T, 1}, Reactant.ConcretePJRTArray{T, 2},
    Reactant.ConcreteIFRTArray{T, 1}, Reactant.ConcreteIFRTArray{T, 2},
}

"""
    CHECKS_PER_LAUNCH

Number of termination checks per launch of a host-driven solve, or `0` (the default) to
choose it adaptively with [`next_checks_per_launch`](@ref).

The chunk size is a runtime argument of the compiled launch, so changing it does not
recompile.
"""
const CHECKS_PER_LAUNCH = Ref(0)

"""
    TARGET_LAUNCH_SECONDS

Wall time that an adaptive launch aims for. Launch overhead is amortized over that long, and
the time limit is overshot by about that much at most.
"""
const TARGET_LAUNCH_SECONDS = Ref(0.02)

"Upper bound on the adaptive number of checks per launch."
const MAX_CHECKS_PER_LAUNCH = 1000

"""
    next_checks_per_launch(checks, launch_seconds, remaining_seconds)

Choose the number of checks of the next launch, from the last one, which ran `checks` checks
in `launch_seconds`.

Aims for launches of [`TARGET_LAUNCH_SECONDS`](@ref), and of at most half the time left before
the time limit, so that the limit is overshot by little. At most doubles from one launch to the
next, since the first launches are the noisiest.
"""
function next_checks_per_launch(checks::Int, launch_seconds::Float64, remaining_seconds::Float64)
    fixed = CHECKS_PER_LAUNCH[]
    fixed > 0 && return fixed
    per_check = max(launch_seconds / checks, 1.0e-9)
    target = min(TARGET_LAUNCH_SECONDS[], remaining_seconds / 2)
    wanted = floor(Int, min(target / per_check, Float64(MAX_CHECKS_PER_LAUNCH)))
    return clamp(wanted, 1, min(2checks, MAX_CHECKS_PER_LAUNCH))
end

"""
    LAUNCH_CACHE

Compiled launches of host-driven solves, keyed by everything that tracing bakes in: the
argument types, the array shapes and devices, and the structural parameters (see
[`collect_shapes!`](@ref)). The number of checks is traced, so it is not part of the key.

It keeps at most `LAUNCH_CACHE_LIMIT[]` executables, evicting the least recently used one,
so that a long session solving many shapes does not keep every compiled program alive.
"""
const LAUNCH_CACHE = Dict{Any, Any}()
"Keys of [`LAUNCH_CACHE`](@ref), least recently used first."
const LAUNCH_CACHE_ORDER = Any[]
"Largest number of compiled launches kept in [`LAUNCH_CACHE`](@ref)."
const LAUNCH_CACHE_LIMIT = Ref(32)
const LAUNCH_CACHE_LOCK = ReentrantLock()

"""
    run_cached_launch!(compile, key, args...)

Call the compiled launch of `key` on `args`, compiling it with `compile()` on a miss (see
[`LAUNCH_CACHE`](@ref)), and return `nothing`. The launch is called here rather than returned,
since the cache holds executables of many types.
"""
function run_cached_launch!(compile, key, args...)
    launch = @lock LAUNCH_CACHE_LOCK begin
        i = findfirst(isequal(key), LAUNCH_CACHE_ORDER)
        if i === nothing
            while length(LAUNCH_CACHE_ORDER) >= max(LAUNCH_CACHE_LIMIT[], 1)
                delete!(LAUNCH_CACHE, popfirst!(LAUNCH_CACHE_ORDER))
            end
            LAUNCH_CACHE[key] = compile()
        else
            deleteat!(LAUNCH_CACHE_ORDER, i)
        end
        push!(LAUNCH_CACHE_ORDER, key)
        LAUNCH_CACHE[key]
    end
    launch(args...)
    return nothing
end

"Copy of `algo` without a time limit, which the host-driven solve enforces itself."
function without_time_limit(algo::CoolPDLP.Algorithm{A}) where {A}
    return CoolPDLP.Algorithm{A}(
        algo.conversion, algo.preconditioning, algo.step_size, algo.restart,
        CoolPDLP.GenericParameters(;
            algo.generic.show_progress, algo.generic.check_every,
            algo.generic.record_error_history, time_limit = Inf,
        ),
        algo.termination,
    )
end

"""
    placement(x)

Return the devices holding the buffers of the device array `x`, which a compiled program is
bound to: two arrays of the same shape on different devices (or sharded differently) need
different executables.
"""
placement(x::Reactant.ConcretePJRTArray) = map(Reactant.XLA.device, x.data)
placement(x::Reactant.ConcreteIFRTArray) = Reactant.XLA.device(x)

"""
    collect_shapes!(acc, x, baked=false)

Push onto `acc` what tracing `x` bakes into a compiled program besides types: the size and
the placement (see [`placement`](@ref)) of every device array, and the plain numbers of the structures traced without number tracking
(the dimensions of a sparse matrix). The clock is left out, since a launch never reads it.
"""
function collect_shapes!(acc::Vector{Any}, x, baked::Bool = false)
    if x isa Union{Reactant.ConcretePJRTArray, Reactant.ConcreteIFRTArray}
        push!(acc, (size(x), placement(x)))
    elseif x isa CoolPDLP.HostClock || x isa Array
        # host-only data, which tracing leaves alone
    elseif x isa Number
        baked && push!(acc, x)
    elseif isstructtype(typeof(x))
        inner = baked || x isa CoolPDLP.GPUSparseMatrix
        for i in 1:nfields(x)
            isdefined(x, i) && collect_shapes!(acc, getfield(x, i), inner)
        end
    end
    return nothing
end

"""
    launch_checks!(state, milp, algo, max_passes)

Run the compiled `run_checks!` for these arguments, compiling it on first use, and return
whether the solve has stopped, read from its termination status, which waits for the launch
to finish.
"""
function launch_checks!(state, milp, algo, max_passes::Reactant.ConcreteRNumber{Int32})
    shapes = Any[]
    collect_shapes!(shapes, (state, milp, algo))
    key = (typeof(state), typeof(milp), typeof(algo), algo.generic, shapes)
    run_cached_launch!(key, state, milp, algo, max_passes) do
        Reactant.@compile CoolPDLP.run_checks!(state, milp, algo, max_passes)
    end
    return CoolPDLP.termination_status(state.stats) != CoolPDLP.MOI.OPTIMIZE_NOT_CALLED
end

"""
    host_driven_solve!(state, milp, algo)

Solve with arrays that already live on a Reactant device, enforcing the time limit on the host.

Each launch runs a number of termination checks on the device (see
[`next_checks_per_launch`](@ref)), with no time limit. The host reads from the termination
status whether they decided to stop, updates the clock in `state.stats`, and either launches
again or stops with `MOI.TIME_LIMIT`. Nothing about time enters the compiled program, and no
host callback is needed, so this works on every Reactant backend, including those where
`@compile solve!` cannot read the clock (AArch64, Metal).
"""
function host_driven_solve!(state, milp, algo)
    launch_algo = without_time_limit(algo)
    (; time_limit) = algo.generic
    stats = state.stats
    checks = CHECKS_PER_LAUNCH[] > 0 ? CHECKS_PER_LAUNCH[] : 1
    while true
        max_passes = Reactant.ConcreteRNumber(Int32(checks))
        launch_start = time()
        must_terminate = launch_checks!(state, milp, launch_algo, max_passes)
        now = time()
        stats.time_elapsed = now - stats.starting_time
        must_terminate && break
        if stats.time_elapsed >= time_limit
            code = CoolPDLP.status_code(CoolPDLP.MOI.TIME_LIMIT)
            stats.termination_status_code = Reactant.ConcreteRNumber(code)
            break
        end
        checks = next_checks_per_launch(
            checks, now - launch_start, Float64(time_limit - stats.time_elapsed)
        )
    end
    return state
end

# Strictly more specific than the solvers' own `solve!`, and only for arrays outside of a
# compilation: under `@compile`, the arrays are traced and the ordinary loop is traced instead.
function CoolPDLP.solve!(
        state::CoolPDLP.PDHGState{T, V}, milp::CoolPDLP.MILP, algo::CoolPDLP.Algorithm{:PDHG}
    ) where {T <: Number, V <: ConcreteRVecOrMat{T}}
    return host_driven_solve!(state, milp, algo)
end
function CoolPDLP.solve!(
        state::CoolPDLP.PDLPState{T, V}, milp::CoolPDLP.MILP, algo::CoolPDLP.Algorithm{:PDLP}
    ) where {T <: Number, V <: ConcreteRVecOrMat{T}}
    return host_driven_solve!(state, milp, algo)
end

# Reactant's generic `mul!` overlays cannot serve a sparse format: they recurse until the stack
# overflows. These are strictly more specific than each of them (same bounds on `c` and `b`, a
# narrower `A`), so they are the ones selected. They call `sparse_mul!` rather than `mul!`, so
# the format's KernelAbstractions kernels are still launched under Reactant, through its
# `ReactantBackend`, instead of natively (which would nest a `Reactant.@jit` on traced
# arguments). This path needs CUDA.jl loaded, on every backend, because Reactant defines
# `ka_with_reactant` in its CUDA extension.
#
# A batched product launches a kernel over a 2D range, and gets an explicit workgroup size:
# without one, `ka_with_reactant` skips its launch configuration when CUDA.jl is not functional
# (e.g. on a machine without a GPU) and launches a single workgroup over the whole range, which
# leaves every column but the first one wrong from 1024 rows on
# (https://github.com/EnzymeAD/Reactant.jl/issues/3815).
const BATCH_WORKGROUPSIZE = (256, 1)

for (cT, bT, kw) in (
        (:AbstractVector, :AbstractVector, :(())),
        (:AbstractMatrix, :AbstractVecOrMat, :((; workgroupsize = BATCH_WORKGROUPSIZE))),
    )
    @eval begin
        @reactant_overlay function LinearAlgebra.mul!(
                c::$cT, A::CoolPDLP.GPUSparseMatrix, b::$bT, α::Number, β::Number
            )
            return CoolPDLP.sparse_mul!(c, A, b, α, β; $kw...)
        end
        @reactant_overlay function LinearAlgebra.mul!(c::$cT, A::CoolPDLP.GPUSparseMatrix, b::$bT)
            return CoolPDLP.sparse_mul!(c, A, b, true, false; $kw...)
        end
    end
end

"""
    CoolPDLP.check_time_limit!(stats, time_limit)

Decide on the host whether a compiled solve has reached its time limit.

Times are `Float64`, which some accelerators (Metal, TPU) cannot hold, so none of them enters
the compiled program: the starting time, the elapsed time and the limit stay on the host, and a
`Reactant.Ops.julia_callback` returns a single `UInt8` flag at each termination check. With an
infinite limit, no callback is emitted at all.

`has_side_effect = true` marks that call impure, so the compiler may not hoist it out of the
loop, share it across iterations or drop it, each of which would freeze the clock.

Where a host callback cannot run, the limit is reported as never reached, with a warning.
"""
@reactant_overlay function CoolPDLP.check_time_limit!(
        stats::CoolPDLP.ConvergenceStats, time_limit::Number
    )
    return traced_time_limit_check(stats, time_limit)
end

end
