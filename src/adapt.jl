const FloatOrFloatArray = Union{AbstractFloat, AbstractArray{<:AbstractFloat}}
const IntOrIntArray = Union{Integer, AbstractArray{<:Integer}}
const NotFloatOrInteger = Union{AbstractString, AbstractArray{<:AbstractString}}

"""
    change_floating_type(T, stuf)

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

function change_floating_type(::Type{T}, A::DeviceSparseMatrixCSR) where {T}
    return DeviceSparseMatrixCSR(
        A.m,
        A.n,
        A.rowptr,
        A.colval,
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

function change_integer_type(::Type{T}, A::DeviceSparseMatrixCSR) where {T}
    return DeviceSparseMatrixCSR(
        A.m,
        A.n,
        change_integer_type(T, A.rowptr),
        change_integer_type(T, A.colval),
        A.nzval
    )
end

for change_type in (:change_floating_type, :change_integer_type)
    @eval begin
        function $change_type(::Type{T}, milp::MILP) where {T}
            (; c, G, h, A, b, l, u, intvar, varname) = milp
            return MILP(;
                c = $change_type(T, c),
                G = $change_type(T, G),
                h = $change_type(T, h),
                A = $change_type(T, A),
                b = $change_type(T, b),
                l = $change_type(T, l),
                u = $change_type(T, u),
                intvar,
                varname
            )
        end

        function $change_type(::Type{T}, sad::SaddlePointProblem) where {T}
            (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad
            return SaddlePointProblem(;
                c = $change_type(T, c),
                q = $change_type(T, q),
                K = $change_type(T, K),
                Kᵀ = $change_type(T, Kᵀ),
                l = $change_type(T, l),
                u = $change_type(T, u),
                m₁ = $change_type(T, m₁),
                m₂ = $change_type(T, m₂)
            )
        end
    end
end

"""
    to_device(::Type{DSM}, problem)
    to_device(problem)

Convert the sparse matrices inside `problem` to a device-friendly matrix format `DSM` from DeviceSparseArrays.jl (the default format is `DeviceSparseMatrixCSR`).

The resulting problem can then be `adapt`-ed to GPU thanks to Adapt.jl.
"""
function to_device end

to_device(problem::AbstractProblem) = to_device(DeviceSparseMatrixCSR, problem)

function to_device(::Type{DSM}, milp::MILP) where {DSM <: AbstractDeviceSparseMatrix}
    (; c, G, h, A, b, l, u, intvar, varname) = milp
    return MILP(; c, G, h, A = DSM(A), b, l, u, intvar, varname)
end

function to_device(::Type{DSM}, sad::SaddlePointProblem) where {DSM <: AbstractDeviceSparseMatrix}
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad
    return SaddlePointProblem(; c, q, K = DSM(K), Kᵀ = DSM(Kᵀ), l, u, m₁, m₂)
end

function Adapt.adapt_structure(to, milp::MILP)
    (; c, G, h, A, b, l, u, intvar, varname) = milp
    return MILP(;
        c = adapt(to, c),
        G = adapt(to, G),
        h = adapt(to, h),
        A = adapt(to, A),
        b = adapt(to, b),
        l = adapt(to, l),
        u = adapt(to, u),
        intvar = adapt(to, intvar),
        varname
    )
end

function Adapt.adapt_structure(to, sad::SaddlePointProblem)
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad
    return SaddlePointProblem(;
        c = adapt(to, c),
        q = adapt(to, q),
        K = adapt(to, K),
        Kᵀ = adapt(to, Kᵀ),
        l = adapt(to, l),
        u = adapt(to, u),
        m₁ = m₁,
        m₂ = m₂
    )
end
