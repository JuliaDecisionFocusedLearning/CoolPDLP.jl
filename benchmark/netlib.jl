using Pkg
Pkg.activate(@__DIR__)

using Accessors
using CoolPDLP

netlib = list_netlib_instances()

milps = map(first ∘ read_netlib_instance, netlib)

params1 = PDLPParameters(;
    termination_reltol = 1.0e-3,
    time_limit = 30.0,
    enable_restarts = false,
    enable_scaling = false,
    enable_primal_weight = false,
)
params2 = @set params1.enable_restarts = true
params3 = @set params2.enable_scaling = true
params4 = @set params3.enable_primal_weight = true

benchmark_results = run_benchmark(pdlp, milps, [params1, params2, params3, params4])
