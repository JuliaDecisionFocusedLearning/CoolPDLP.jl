using Pkg
Pkg.activate(@__DIR__)

using CoolPDLP
using CSV
using DataFrames
using SparseArrays

instance_folder = joinpath(@__DIR__, "data", "instances")

all_milps = map(1:50) do i
    @info "$i"
    i_str = i < 10 ? "0$i" : "$i"
    read_milp(joinpath(instance_folder, "instance_$i_str.mps"))
end

df = DataFrame(;
    instance = 1:50,
    nbvar = map(nbvar, all_milps),
    nbvar_int = map(nbvar_int, all_milps),
    nbvar_cont = map(nbvar_cont, all_milps),
    nbcons = map(nbcons, all_milps),
    nbcons_eq = map(nbcons_eq, all_milps),
    nbcons_ineq = map(nbcons_ineq, all_milps),
    nonzeros = map(milp -> nnz(milp.A) + nnz(milp.G), all_milps),
    nonzeros_eq = map(milp -> nnz(milp.A), all_milps),
    nonzeros_ineq = map(milp -> nnz(milp.G), all_milps),
)

CSV.write(joinpath(@__DIR__, "summary.csv"), df)
