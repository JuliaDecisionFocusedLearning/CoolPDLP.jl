"""
    SpiralCrossover

Liu–Lu Algo 2: PCG rays ([`ols!`](@ref)) on a host CSC lift in the algorithm
eltype `T`, then a basis. `PDLP(Float32)` stays in `Float32`; the default
`PDLP()` stays in `Float64`. `solve` calls it only when the status is
`OPTIMAL` and `time_limit - time_elapsed > 0`. That remainder is one budget
for every ray. Relative stops use `termination_reltol`. Comparisons to zero
use `zero_tol`. A finite value is kept even when the primal residual grew.
A non-finite value keeps the PDLP point.
`max_pushes` (default `10_000`) is a fuse on the ratio-test loop. The loop
stops when the basic count stops falling.
The basis is one column-pivoted QR. The square cleanup is the dense solve `\\`.
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
