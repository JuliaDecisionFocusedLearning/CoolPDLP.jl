using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Adapt
using CoolPDLP
using KernelAbstractions
using Metal

params = PDHGParameters(;
    tol_termination = 1.0f-3, max_kkt_passes = 10^2, check_every = 10^2
)

netlib = list_netlib_instances(; exclude_failing = true)
milp = read_netlib_instance(netlib[1])[1]

sad = SaddlePointProblem(milp)
sad_32 = single_precision(sad)
sad_32_csr = change_matrix_type(DeviceSparseMatrixCSR, sad_32)
sad_32_csr_metal = adapt(MetalBackend(), sad_32_csr)

@time pdhg(sad_32, params; show_progress = true)
@time pdhg(sad_32_csr, params; show_progress = true)
@time Metal.@sync pdhg(sad_32_csr_metal, params; show_progress = false)
