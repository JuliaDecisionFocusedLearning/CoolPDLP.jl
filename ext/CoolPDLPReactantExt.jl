module CoolPDLPReactantExt

using CoolPDLP: CoolPDLP
using Reactant: Reactant, @reactant_overlay

"""
    write_time!(out)

Write the current host time into the single-element output buffer of a Reactant callback.

`Reactant.Ops.julia_callback` hands the callback its output buffers first and its inputs
afterwards, and a `()`-shaped output arrives dereferenced, as a plain `Float64` with nothing to
write into. The output is therefore declared with shape `(1,)` and reduced back to a scalar on
the traced side.
"""
write_time!(out::AbstractVector{Float64}) = (out[1] = time(); nothing)

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
    out = Reactant.Ops.julia_callback(
        write_time!, ((Float64, (1,)),); has_side_effect = true
    )
    return sum(out)
end

end
