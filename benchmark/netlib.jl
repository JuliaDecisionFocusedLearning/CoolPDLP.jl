using Pkg
Pkg.activate(@__DIR__)

using Accessors
using CairoMakie
using CoolPDLP

netlib = list_netlib_instances()

milps = map(read_netlib_instance, netlib)

params1 = PDLPParameters(;
    termination_reltol = 1.0e-2,
    time_limit = 10.0,
    enable_restarts = false,
    enable_scaling = false,
    enable_primal_weight = false,
)
params2 = @set (@set params1.enable_restarts = true)
params3 = @set params2.enable_scaling = true
params4 = @set params3.enable_primal_weight = true

params1_renamed = @set params1.name = "PDHG"
params2_renamed = @set params2.name = "+ restarts"
params3_renamed = @set params3.name = "+ scaling"
params4_renamed = @set params4.name = "+ primal weight"

params_candidates = [
    params1_renamed,
    params2_renamed,
    params3_renamed,
    params4_renamed,
]

benchmark_results = run_benchmark(pdlp, milps, params_candidates)

function plot_profiles(benchmark_results)
    common_reltol = only(unique(map(r -> r.params.termination_reltol, benchmark_results)))
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "KKT passes",
        ylabel = "Fraction of problems solved to $common_reltol",
        title = "Convergence of PDLP",
    )
    for result in benchmark_results
        (; params, kkt_passes) = result
        scatterlines!(
            ax,
            vcat(0, sort(kkt_passes)),
            (0:length(milps)) ./ length(milps),
            label = params.name
        )
    end
    axislegend(ax, position = :lt)
    return fig
end

plot_profiles(reverse(results))
