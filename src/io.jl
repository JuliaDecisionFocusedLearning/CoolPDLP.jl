"""
    read_milp(path::String)

Read an optimization problem stored in a (possibly gzipped) MPS file stored at `path`, return an [`MILP`](@ref) object.
"""
function read_milp(path::String)
    if endswith(path, ".mps.gz")
        contents = GZip.open(path, "r") do f
            read(f, String)
        end
        mps_path = tempname(; suffix = ".mps")
        open(mps_path, "w") do f
            write(f, contents)
        end
    elseif endswith(path, ".mps")
        mps_path = path
    end

    qps_data = with_logger(NullLogger()) do
        readqps(mps_path)
    end
    (; arows, acols, avals, lcon, ucon, lvar, uvar, c, vartypes, varnames) = qps_data

    A_eq_ineq = sparse(arows, acols, avals)
    L = lcon
    U = ucon
    l = lvar
    u = uvar
    c = qps_data.c

    eq_inds = L .== U
    larger_inds = .!eq_inds .& (L .> typemin(eltype(L)))
    smaller_inds = .!eq_inds .& (U .< typemax(eltype(L)))

    A = A_eq_ineq[eq_inds, :]
    b = U[eq_inds]

    G = vcat(A_eq_ineq[larger_inds, :], -A_eq_ineq[smaller_inds, :])
    h = vcat(L[larger_inds], -U[smaller_inds])

    binvar = vartypes .== VTYPE_Binary
    intvar = vartypes .== VTYPE_Integer
    @assert all(0 .== l[binvar])
    @assert all(1 .== u[binvar])

    varname = varnames

    milp = MILP(; c, G, h, A, b, l, u, intvar = binvar .| intvar, varname)
    return milp
end


"""
    read_sol(path::String, milp::MILP)

Read a solution stored in a `.sol` file following the MIPLIB specification, return a vector of floating-point numbers.
"""
function read_sol(path::String, milp::MILP)
    T = eltype(milp)
    x = fill(convert(T, NaN), nbvar(milp))
    open(path, "r") do f
        obj = NaN
        i = 1
        for line in eachline(f)
            if isempty(line)
                continue
            elseif startswith(line, "=obj=")
                obj = parse(T, split(line)[end])
            else
                v = parse(T, split(line)[end])  # TODO: handle more generic separators?
                x[i] = v
                i += 1
            end
        end
        @assert objective_value(x, milp) ≈ obj
    end
    return x
end


"""
    write_sol(path::String, x::AbstractVector, milp::MILP)

Write a solution to a `.sol` file following the MIPLIB specification.
"""
function write_sol(path::String, x::AbstractVector{<:Number}, milp::MILP)
    @assert endswith(path, ".sol")
    x = float.(x)
    open(path, "w") do f
        print(f, "=obj= ")
        print(f, objective_value(x, milp))
        ret = "\n"
        space = " "
        for i in eachindex(x)
            print(f, ret, milp.varname[i], space, x[i])
        end
    end
    return
end
