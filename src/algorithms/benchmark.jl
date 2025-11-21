"""
    BenchmarkResult

Store the results of a solver over several instances.
"""
@kwdef struct BenchmarkResult{P, S}
    params::P
    states::Vector{S}
    kkt_passes::Vector{Int}
    time_elapsed::Vector{Float64}
    kkt_passes_sgm10::Float64
    time_elapsed_sgm10::Float64
end

function run_benchmark(
        solver::S, milps::Vector{<:MILP}, params::AbstractParameters
    ) where {S <: Function}
    prog = Progress(length(milps); desc = "Benchmarking $solver:")
    states = tmap(milps) do milp
        next!(prog)
        solver(milp, params; show_progress = false)[end]
    end
    kkt_passes = map(states) do state
        if state.termination_reason == CONVERGENCE
            state.kkt_passes
        else
            params.max_kkt_passes
        end
    end
    time_elapsed = map(states) do state
        if state.termination_reason == CONVERGENCE
            state.time_elapsed
        else
            params.time_limit
        end
    end
    kkt_passes_sgm10 = exp(mean(log.(kkt_passes .+ 10))) .- 10
    time_elapsed_sgm10 = exp(mean(log.(time_elapsed .+ 10))) .- 10
    return BenchmarkResult(;
        params, states, kkt_passes, time_elapsed, kkt_passes_sgm10, time_elapsed_sgm10
    )
end

function run_benchmark(
        solver::S, milps::Vector{<:MILP}, params_candidates::Vector{<:AbstractParameters}
    ) where {S}
    results = map(params_candidates) do params
        @info "Benchmarking" params
        run_benchmark(solver, milps, params)
    end
    return results
end
