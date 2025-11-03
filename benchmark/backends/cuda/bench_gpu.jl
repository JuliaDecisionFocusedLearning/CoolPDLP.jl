using Pkg
Pkg.activate(joinpath(@__DIR__, "backends", raw"cuda"))

using CoolPDLP
using KernelAbstractions
using CUDA

params = PDHGParameters(; tol_termination = 1.0e-3, max_kkt_passes = 10^2, check_every = 10^2)

netlib = list_netlib_instances(; exclude_failing = true)
milp = read_netlib_instance(netlib[1])[1]

milp_32 = change_integer_type(Int32, change_floating_type(Float32, milp))
milp_cuda = adapt(CUDABackend(), to_device(milp_32))

@time pdhg(milp_32, params; show_progress = true)
CUDA.@time pdhg(milp_cuda, params; show_progress = false)
