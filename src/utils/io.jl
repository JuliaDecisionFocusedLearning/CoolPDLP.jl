"""
    read_sol(path::String, milp::MILP)

Read a solution stored in a `.sol` file following the MIPLIB specification, return a vector of floating-point numbers.
"""
function read_sol(path::String, milp::MILP)
    T = Float64
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
            print(f, ret, milp.var_name[i], space, x[i])
        end
    end
    return
end
