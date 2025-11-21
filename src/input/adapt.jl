function common_backend(arrs::Vararg{Any,N}) where {N}
    backends = map(get_backend, arrs)
    @assert all(==(backends[1]), backends)
    return backends[1]
end

function KernelAbstractions.get_backend(sad::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    return common_backend(c, q, K, Kᵀ, l, u, ineq_cons)
end

const FloatOrFloatArray = Union{AbstractFloat,AbstractArray{<:AbstractFloat}}
const IntOrIntArray = Union{Integer,AbstractArray{<:Integer}}
const NotFloatOrInteger = Union{AbstractString,AbstractArray{<:AbstractString}}

"""
    set_eltype(T, stuff)

Change the element type of floating-point containers inside `stuff` to `T`.
"""
function set_eltype end

set_eltype(::Type{T}, A::FloatOrFloatArray) where {T} = map(T, A)
set_eltype(::Type{T}, A::Union{IntOrIntArray,NotFloatOrInteger}) where {T} = A

function set_eltype(::Type{T}, A::SparseMatrixCSC) where {T}
    return SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, set_eltype(T, A.nzval))
end

"""
    set_indtype(T, stuff)

Change the element type of integer containers inside `stuff` to `T`.
"""
function set_indtype end

set_indtype(::Type{T}, A::IntOrIntArray) where {T} = map(T, A)
set_indtype(::Type{T}, A::Union{FloatOrFloatArray,NotFloatOrInteger}) where {T} = A

function set_indtype(::Type{T}, A::SparseMatrixCSC) where {T}
    return SparseMatrixCSC(
        A.m, A.n, set_indtype(T, A.colptr), set_indtype(T, A.rowval), A.nzval
    )
end

for change_type in (:set_eltype, :set_indtype)
    @eval begin
        function $change_type(::Type{T}, sad::SaddlePointProblem) where {T}
            (; c, q, K, Kᵀ, l, u, ineq_cons, preconditioner) = sad
            return SaddlePointProblem(;
                c=$change_type(T, c),
                q=$change_type(T, q),
                K=$change_type(T, K),
                Kᵀ=$change_type(T, Kᵀ),
                l=$change_type(T, l),
                u=$change_type(T, u),
                ineq_cons,
                preconditioner,
            )
        end
    end
end

"""
    single_precision(problem)

Convert all integers in `problem` to `Int32` and all floating-point numbers to `Float32`.
"""
single_precision(x) = set_eltype(Float32, set_indtype(Int32, x))

"""
    set_matrix_type(::Type{M}, problem)

Convert the sparse matrices inside `problem` using constructor `M`.
"""
function set_matrix_type end

function set_matrix_type(::Type{M}, sad::SaddlePointProblem) where {M<:AbstractMatrix}
    (; c, q, K, Kᵀ, l, u, ineq_cons, preconditioner) = sad
    return SaddlePointProblem(; c, q, K=M(K), Kᵀ=M(Kᵀ), l, u, ineq_cons, preconditioner)
end

function Adapt.adapt_structure(to, sad::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, ineq_cons, preconditioner) = sad
    return SaddlePointProblem(;
        c=adapt(to, c),
        q=adapt(to, q),
        K=adapt(to, K),
        Kᵀ=adapt(to, Kᵀ),
        l=adapt(to, l),
        u=adapt(to, u),
        ineq_cons=adapt(to, ineq_cons),
        preconditioner,
    )
end
