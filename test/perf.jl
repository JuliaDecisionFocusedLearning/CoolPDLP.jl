using Chairmarks
using CoolPDLP
using MathOptBenchmarkInstances
using ProgressMeter
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
