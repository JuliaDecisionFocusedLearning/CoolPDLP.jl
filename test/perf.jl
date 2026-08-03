using Chairmarks
using CoolPDLP
using CoolPDLP: nbinstances, restart!, restart_check!, step!, termination_check!
using MathOptBenchmarkInstances
using ProgressMeter
using Random
using SparseArrays
using Test

prepstate(milp, algo) = initialize(
    milp, PrimalDualSolution(milp), algo; starting_time = time()
)

@testset verbose = true "Allocation-free `solve!`" begin
    milp = MILP(read_instance(Netlib, first(list_instances(Netlib)))[1])
    @testset "$(typeof(algo))" for algo in [
            PDHG(time_limit = 1.0, record_error_history = false)
            PDLP(time_limit = 1.0, record_error_history = false)
        ]
        milp = MILP(read_instance(Netlib, first(list_instances(Netlib)))[1])
        algo = PDHG(time_limit = 1.0, record_error_history = false)
        solve!(prepstate(milp, algo), milp, algo)
        result = @b prepstate(milp, algo) solve!(_, milp, algo) seconds = 5
        result_nosolve = @b ProgressUnknown(; desc = "placeholder")
        @test result.allocs == result_nosolve.allocs
    end
end

function run_iterations!(state, milp, algo, n)
    for _ in 1:n
        step!(state, milp)
        termination_check!(state, milp, algo)
        restart_check!(state, milp, algo) && restart!(state, algo)
    end
    return nothing
end

function iteration_allocations(state, milp, algo)
    run_iterations!(state, milp, algo, 5)
    return @allocated run_iterations!(state, milp, algo, 20)
end

@testset verbose = true "Allocation-free iterations" begin
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, 3; batched = filter(!=(:A), BATCHABLE))

    algo = PDLP(; record_error_history = false)
    @testset "batch size $(nbinstances(milp))" for milp in (milps[1], milp_batch)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        @test iteration_allocations(state, milp, algo) == 0
    end
end
