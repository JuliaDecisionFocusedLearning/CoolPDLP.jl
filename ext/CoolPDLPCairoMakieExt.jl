module CoolPDLPCairoMakieExt

using CoolPDLP
using CairoMakie

function CoolPDLP.plot_profiles(benchmark_results)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "KKT passes",
        ylabel = "Fraction of problems solved",
        title = "Convergence of PDLP",
    )
    for (k, result) in benchmark_results
        (; states, kkt_passes, time_elapsed) = result
        scatterlines!(ax, sort(kkt_passes), 1:length(milps), label = "option $k")
    end
    axislegend(ax, position = :rb)
    return fig
end

end
