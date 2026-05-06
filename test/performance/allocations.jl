using CoolPDLP
using Chairmarks
using MathOptBenchmarkInstances
using SparseArrays
using Test

@testset verbose = true "Allocation-free `solve!`" begin
    milp = MILP(read_instance(Netlib, first(list_instances(Netlib)))[1])
    @testset "$(typeof(algo))" for algo in [
            PDHG(time_limit = 1.0, record_error_history = false)
            PDLP(time_limit = 1.0, record_error_history = false)
        ]
        prepstate() = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        result = @b prepstate() solve!(_, milp, algo) seconds = 10
        @test result.allocs == 0
    end
end
