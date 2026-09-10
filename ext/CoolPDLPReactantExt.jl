module CoolPDLPReactantExt

using CoolPDLP: CoolPDLP
using LinearAlgebra: LinearAlgebra
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay

"""
    CoolPDLP.batched_bool_type(v)

Give the type of the boolean that reducing a traced batch yields.

Reactant types every reduction over a `TracedRArray` as `Union{TracedRArray, TracedRNumber}`,
accurate enough to trace but too coarse for the callers: the abstract type would escape into
the return type of the termination and restart checks of a batched solve. The reduced value is
in fact a traced scalar, which is what this returns.
Once https://github.com/EnzymeAD/Reactant.jl/issues/3261 is solved upstream, this can be removed.
"""
CoolPDLP.batched_bool_type(::TracedRArray) = TracedRNumber{Bool}

# Trace one of CoolPDLP's own sparse formats without turning its shape into a runtime input.
#
# `m` and `n` are plain `Int` fields, and any mode that promotes Julia numbers -- `to_rarray(…;
# track_numbers = true)`, but also the `@trace while` loop of `solve!`, which promotes the numbers
# of everything the loop carries -- derives `TracedRNumber{Int64}` for them. The struct declares
# them as `Int`, so Reactant cannot express the converted type and gives up with a
# `NoFieldMatchError`. Moving the shape into the type parameters would fix that, at the price of a
# distinct Julia type (and so a fresh compilation of the whole solver) per matrix shape.
#
# The shape of a sparse matrix is structure rather than data, and so is its sparsity pattern: no
# solve ever wants either as a runtime scalar. Both overloads below say exactly that, by tracing
# these wrappers with number tracking switched off. Their nonzero values and index arrays are
# arrays, so they are still traced as usual.
function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:CoolPDLP.GPUSparseMatrix}),
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
        @nospecialize(prev::CoolPDLP.GPUSparseMatrix),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        kwargs...,
    )
    return Reactant.make_tracer_unknown(
        seen, prev, path, mode; track_numbers = Union{}, kwargs...
    )
end

# Send a product by one of CoolPDLP's own sparse formats back to its kernel.
#
# Reactant overlays `mul!` for every `AbstractMatrix` and lowers the product to a dense
# `stablehlo.dot_general`. An overlay takes precedence over any ordinary method, so without a
# more specific overlay the kernel-backed `mul!` of these formats is never reached: the overlay
# tries to materialise the matrix as a traced array instead, and since such a wrapper is an
# `AbstractArray{<:TracedRNumber}` that is *not* a view of a `TracedRArray`, Reactant's
# `get_ancestor_and_indices` recurses on it until the stack overflows.
#
# `CoolPDLP.spmul!` is the same kernel launch under a name Reactant does not overlay. The launch
# itself still goes through Reactant's own `KernelAbstractions` overlay, which is what turns it
# into a GPU kernel inside the compiled program.
#
# The two shapes mirror Reactant's own overlays, so that each of these is strictly more specific
# than the one it has to beat.
@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector, α::Number, β::Number
    )
    return CoolPDLP.spmul!(c, A, b, α, β)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractVecOrMat, α::Number, β::Number
    )
    return spmul_batched!(c, A, b, α, β)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector
    )
    return CoolPDLP.spmul!(c, A, b, true, false)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractVecOrMat
    )
    return spmul_batched!(c, A, b, true, false)
end

"""
    spmul_batched!(c, A, b, α, β)

A batched sparse product inside a compiled program, run on flattened operands.

This works around a miscompilation on Reactant's CUDA backend: a 2-D array written by a
`KernelAbstractions` kernel inside a `@trace for` loop and reduced after the loop comes back
wrong. XLA's layout assignment lets the reduction pick a row-major layout for the loop-carried
array, and since the `enzymexla.kernel_call` pins neither its operand nor its result layouts,
XLA hands the kernel's column-major bytes over as if they were row-major. Nothing placed between
the loop and the reduction helps (`copy`, a broadcast, an `optimization_barrier` -- the
preference propagates through them all), and writing the kernel's output into a fresh 2-D array
gets undone by XLA's copy elision in a program of this size.

A 1-D array has a single layout, and `Reactant.Ops.reshape` is an op XLA has to honour, so the
kernel here reads a reshaped copy of `b`, writes a fresh 1-D buffer, and `c` is filled from its
reshape. The vector case has nothing to fix and keeps the direct launch. The proper fix belongs
upstream: the lowering of `enzymexla.kernel_call` should carry `operand_layouts` *and*
`result_layouts`, as `Reactant.Ops.julia_callback` already does for its own custom call.
"""
function spmul_batched!(c::AbstractMatrix, A, b::AbstractMatrix, α::Number, β::Number)
    m, nb = size(c)
    b_flat = Reactant.Ops.reshape(b, length(b))
    tmp = similar(c, length(c))
    # the kernel reads its destination whenever it accumulates into it
    (β isa TracedRNumber || !iszero(β)) && (tmp .= Reactant.Ops.reshape(c, length(c)))
    CoolPDLP.spmm!(tmp, A, b_flat, nb, α, β)
    c .= Reactant.Ops.reshape(tmp, m, nb)
    return c
end

"""
    write_time!(out)

Write the current host time into the single-element output buffer of a Reactant callback.

`Reactant.Ops.julia_callback` hands the callback its output buffers first and its inputs
afterwards, and a `()`-shaped output arrives dereferenced, as a plain `Float64` with nothing to
write into. The output is therefore declared with shape `(1,)` and reduced back to a scalar on
the traced side.

The write goes through `fill!` rather than `out[1] =`: on the CUDA backend the buffer handed to
the callback lives on the device, and CUDA.jl refuses a scalar `setindex!` on a GPU array. The
refusal is raised inside Reactant's trampoline, which turns it into
`reactant_julia_callback: callback returned false` at run time, long after tracing. `fill!`
writes the single element on any backend.
"""
write_time!(out::AbstractVector{Float64}) = (fill!(out, time()); nothing)

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
    This Reactant backend cannot run a host callback, so the elapsed time inside a compiled \
    solve stays frozen at its compilation-time value and `time_limit` will not be enforced. \
    Loading CUDA.jl lifts this on the CUDA backend.""" maxlog = 1
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
