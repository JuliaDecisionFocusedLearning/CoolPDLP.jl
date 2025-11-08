using Pkg
Pkg.activate(@__DIR__)

using Accessors
using CairoMakie
using CoolPDLP
using Statistics

Base.get_extension(CoolPDLP, :CoolPDLPCairoMakieExt)

netlib = list_netlib_instances()

milps = map(first ∘ read_netlib_instance, netlib)

params1 = PDLPParameters(;
    termination_reltol = 1.0e-2,
    time_limit = 10.0,
    max_kkt_passes = 10^3
)
params2 = @set params1 enable_restarts = true
params3 = @set params2 enable_scaling = true
params4 = @set params3 enable_primal_weight = true

states = tmap(names_and_milps) do (name, milp)
    @info "$name"
    pdhg(milp, params; show_progress = false)
end

function plot_profile(states, params)
    max_time = maximum(s -> s.time_elapsed, states)
    time_range = range(0, max_time, 100)
    fractions = [
        mean(s -> s.time_elapsed <= t && s.termination_reason == CoolPDLP.CONVERGENCE, states)
            for t in time_range
    ]

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Wall clock time (secs)",
        ylabel = "Fraction of problems solved",
        title = "Convergence of LP solvers on Netlib",
        subtitle = "Tolerance - $(params.termination_reltol)"
    )
    lines!(ax, time_range, fractions; label = "baseline PDHG")
    axislegend(position = :rb)
    return fig
end

plot_profile(states, params)
