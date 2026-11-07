module CoolPDLPReactantExt

using CoolPDLP: CoolPDLP
using LinearAlgebra: LinearAlgebra
using Reactant: Reactant, @reactant_overlay

# A sparse matrix's shape and pattern are structure, and so are the generic parameters: with a
# traced `check_every`, the inner loop of `solve!` has a dynamic trip count, and XLA reads its
# condition back to the host after every KKT pass. Trace them without number tracking.
const Structural = Union{CoolPDLP.GPUSparseMatrix, CoolPDLP.GenericParameters}

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

# Reactant's generic `mul!` overlays cannot serve a sparse format: they recurse until the stack
# overflows. These are strictly more specific than each of them (same bounds on `c` and `b`, a
# narrower `A`), so they are the ones selected. They call `sparse_mul!` rather than `mul!`, so
# the format's KernelAbstractions kernels are still launched under Reactant, through its
# `ReactantBackend`, instead of natively (which would nest a `Reactant.@jit` on traced
# arguments). This path needs CUDA.jl loaded, on every backend, because Reactant defines
# `ka_with_reactant` in its CUDA extension.
for (cT, bT) in ((:AbstractVector, :AbstractVector), (:AbstractMatrix, :AbstractVecOrMat))
    @eval begin
        @reactant_overlay function LinearAlgebra.mul!(
                c::$cT, A::CoolPDLP.GPUSparseMatrix, b::$bT, α::Number, β::Number
            )
            return CoolPDLP.sparse_mul!(c, A, b, α, β)
        end
        @reactant_overlay function LinearAlgebra.mul!(c::$cT, A::CoolPDLP.GPUSparseMatrix, b::$bT)
            return CoolPDLP.sparse_mul!(c, A, b, true, false)
        end
    end
end

"""
    write_time!(out)

Write the current host time into the single-element output buffer of a Reactant callback.

`Reactant.Ops.julia_callback` hands the callback its output buffers first and its inputs
afterwards, and a `()`-shaped output arrives dereferenced, as a plain `Float64` with nothing to
write into. The output is therefore declared with shape `(1,)` and reduced back to a scalar on
the traced side.

Uses `fill!` because CUDA.jl refuses `out[1] = ...` on a device buffer.
"""
write_time!(out::AbstractVector{Float64}) = (fill!(out, time()); nothing)

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

"""
    frozen_clock()

Read the clock the old way, for backends that cannot run a host callback.

The value is read once while tracing and baked into the compiled program as a constant, so the
elapsed time never advances and the time limit never fires. Warn rather than fail: everything
else about a compiled solve still works, and the KKT pass budget still bounds it.
"""
function frozen_clock()
    @warn """
    This platform or Reactant backend cannot run a host callback, so the elapsed time inside \
    a compiled solve stays frozen at its compilation-time value and `time_limit` will not be \
    enforced. On an x86_64 machine, loading CUDA.jl lifts this on the CUDA backend.""" maxlog = 1
    return time()
end

"""
    CoolPDLP.current_time()

Read the host clock from inside a compiled program.

Tracing `Base.time()` would freeze its trace-time value into the compiled program as a
constant, so the elapsed time would never advance and the time limit could never fire.
`Reactant.Ops.julia_callback` emits a `stablehlo.custom_call` back into Julia instead, which is
re-evaluated at every iteration of the compiled loop.

`has_side_effect = true` marks that call impure, so the compiler may not hoist it out of the
loop, share it across iterations or drop it when its result looks unused — each of which would
put the frozen clock back. With `has_side_effect = false` the emitted call is pure and all
three become legal.

The single-element reduction that turns the callback's output back into a scalar costs nothing:
it compiles down to a `stablehlo.reshape`.
"""
@reactant_overlay function CoolPDLP.current_time()
    host_callbacks_supported() || return frozen_clock()
    out = Reactant.Ops.julia_callback(
        write_time!, ((Float64, (1,)),); has_side_effect = true
    )
    return sum(out)
end

end
