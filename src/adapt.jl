function common_backend(arrs::Vararg{Any, N}) where {N}
    backends = map(get_backend, arrs)
    @assert all(==(backends[1]), backends)
    return backends[1]
end

function KernelAbstractions.get_backend(sad::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2) = sad
    return common_backend(c, q, K, Kᵀ, l, u, ineq_cons, D1, D2)
end

const FloatOrFloatArray = Union{AbstractFloat, AbstractArray{<:AbstractFloat}}
const IntOrIntArray = Union{Integer, AbstractArray{<:Integer}}
const NotFloatOrInteger = Union{AbstractString, AbstractArray{<:AbstractString}}

"""
    change_floating_type(T, stuff)

Change the element type of floating-point containers inside `stuff` to `T`.
"""
function change_floating_type end

change_floating_type(::Type{T}, A::FloatOrFloatArray) where {T} = map(T, A)
change_floating_type(::Type{T}, A::Union{IntOrIntArray, NotFloatOrInteger}) where {T} = A

function change_floating_type(::Type{T}, A::SparseMatrixCSC) where {T}
    return SparseMatrixCSC(
        A.m,
        A.n,
        A.colptr,
        A.rowval,
        change_floating_type(T, A.nzval)
    )
end

"""
    change_integer_type(T, stuff)

Change the element type of integer containers inside `stuff` to `T`.
"""
function change_integer_type end

change_integer_type(::Type{T}, A::IntOrIntArray) where {T} = map(T, A)
change_integer_type(::Type{T}, A::Union{FloatOrFloatArray, NotFloatOrInteger}) where {T} = A

function change_integer_type(::Type{T}, A::SparseMatrixCSC) where {T}
    return SparseMatrixCSC(
        A.m,
        A.n,
        change_integer_type(T, A.colptr),
        change_integer_type(T, A.rowval),
        A.nzval
    )
end

for change_type in (:change_floating_type, :change_integer_type)
    @eval begin
        function $change_type(::Type{T}, sad::SaddlePointProblem) where {T}
            (; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2) = sad
            return SaddlePointProblem(;
                c = $change_type(T, c),
                q = $change_type(T, q),
                K = $change_type(T, K),
                Kᵀ = $change_type(T, Kᵀ),
                l = $change_type(T, l),
                u = $change_type(T, u),
                ineq_cons,
                D1 = $change_type(T, D1),
                D2 = $change_type(T, D2),
            )
        end
    end
end

"""
    single_precision(problem)

Convert all integers in `problem` to `Int32` and all floating-point numbers to `Float32`.
"""
single_precision(problem) = change_integer_type(Int32, change_floating_type(Float32, problem))

"""
    change_matrix_type(::Type{M}, problem)

Convert the sparse matrices inside `problem` using constructor `M`.
"""
function change_matrix_type end

function change_matrix_type(::Type{M}, sad::SaddlePointProblem) where {M <: AbstractMatrix}
    (; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2) = sad
    return SaddlePointProblem(; c, q, K = M(K), Kᵀ = M(Kᵀ), l, u, ineq_cons, D1, D2)
end

function Adapt.adapt_structure(to, sad::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, ineq_cons, D1, D2) = sad
    return SaddlePointProblem(;
        c = adapt(to, c),
        q = adapt(to, q),
        K = adapt(to, K),
        Kᵀ = adapt(to, Kᵀ),
        l = adapt(to, l),
        u = adapt(to, u),
        ineq_cons = adapt(to, ineq_cons),
        D1 = adapt(to, D1),
        D2 = adapt(to, D2),
    )
end
