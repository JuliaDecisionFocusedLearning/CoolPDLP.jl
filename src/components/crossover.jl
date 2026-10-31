"""
    SpiralCrossover

Liu–Lu Algo 2: least-squares rays ([`ols!`](@ref)) on a host CSC lift in the
algorithm eltype `T`, then a basis. The ray is a QR below 1000 rows or
columns, LSMR up to 5000, and a ridged PCG above that. The undamped solve
stays in the kernel of the active columns; the ridge does not. `PDLP(Float32)`
stays in `Float32`; the default
`PDLP()` stays in `Float64`. `solve` calls it only when the status is
`OPTIMAL` and `time_limit - time_elapsed > 0`. That remainder is one budget
for every ray. Relative stops use `termination_reltol`. Comparisons to zero
use `zero_tol`. The candidate is written when its primal violation is
finite and at most `max(termination_reltol, violation of the incoming
point)`. Otherwise the PDLP point stays, as it does for a non-finite value.
`max_pushes` (default `10_000`) is a fuse on the ratio-test loop. The loop
stops when the basic count stops falling.
A column-pivoted QR checks the columns the pushes left basic. It does not
drop one of them. When that set is short of `m` and still full rank, a second
QR on its orthogonal complement may add dual-active columns. The square
cleanup is the dense solve `\\`, and it
writes `x_B` only when that solution already lies within the bounds.
Both stay on a host block of size `m`. The ray products go through `mul!`,
which is the step a device matrix can replace.
"""
@kwdef struct SpiralCrossover{T <: Real}
    eta::T = one(T)
    omega::T = one(T)
    delta::T = one(T)
    gamma::T = one(T)
    seed::Int = 0
    max_pushes::Int = 10_000
end
