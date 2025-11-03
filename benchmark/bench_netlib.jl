using Pkg
Pkg.activate(@__DIR__)

using Base.Threads
using CairoMakie
using CoolPDLP
using OhMyThreads
using Statistics

@show nthreads()

names_and_milps = [
    (name, read_netlib_instance(name)[1])
        for name in list_netlib_instances(; exclude_failing = true)
]

params = PDHGParameters(; time_limit = 100.0, max_kkt_passes = 10^8)

states = tmap(names_and_milps) do (name, milp)
    @info "$name"
    pdhg(milp, params; show_progress = false)
end

function plot_profile(states, params)
    max_time = maximum(s -> s.elapsed, states)
    time_range = range(0, max_time, 100)
    fractions = [
        mean(s -> s.elapsed <= t && s.termination_reason == CoolPDLP.CONVERGENCE, states)
            for t in time_range
    ]

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Wall clock time (secs)",
        ylabel = "Fraction of problems solved",
        title = "Convergence of LP solvers on Netlib",
        subtitle = "Tolerance - $(params.tol_termination)"
    )
    lines!(ax, time_range, fractions; label = "baseline PDHG")
    axislegend(position = :rb)
    return fig
end

plot_profile(states, params)
