"""
    ols!(x, A, b)

`min ‖A x − b‖₂` by a column-pivoted QR of `A`.

A square singular slice makes sparse `\\` throw `SingularException` (UMFPACK).
The pivoted factor returns a basic solution instead, and a consistent
`b = A c` still puts `c − x` in `ker(A)`.
"""
function ols!(x::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where {T}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("x length $(length(x)) ≠ ncols $n"))
    length(b) == m || throw(DimensionMismatch("b length $(length(b)) ≠ nrows $m"))
    (m == 0 || n == 0) && return x
    rhs = Vector{T}(b)
    copyto!(x, qr(Matrix{T}(A), ColumnNorm()) \ rhs)
    return x
end
