using Chairmarks
using CoolPDLP
using CoolPDLP: restart!, restart_check!, step!, termination_check!
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
    nbatch = 3

    # several problems sharing the same constraint matrix, but with their own objective and bounds
    milps_init = [CoolPDLP.random_milp_and_sol(20, 30, 0.4)[1] for _ in 1:nbatch]
    A, int_var = milps_init[1].A, milps_init[1].int_var
    milps = map(milps_init) do m
        MILP(; c = m.c, lv = m.lv, uv = m.uv, A, lc = m.lc, uc = m.uc, int_var)
    end
    sols = map(PrimalDualSolution, milps)

    stack_batch(f) = reduce(hcat, map(f, milps))
    milp_batch = MILP(;
        c = stack_batch(m -> m.c),
        lv = stack_batch(m -> m.lv),
        uv = stack_batch(m -> m.uv),
        A,
        lc = stack_batch(m -> m.lc),
        uc = stack_batch(m -> m.uc),
        int_var,
    )
    sol_batch = PrimalDualSolution(milp_batch)

    algo = PDLP(; record_error_history = false)
    @testset "batch size $(size(sol.x, 2))" for (milp, sol) in
        ((milps[1], sols[1]), (milp_batch, sol_batch))
        state = initialize(milp, copy(sol), algo; starting_time = time())
        @test iteration_allocations(state, milp, algo) == 0
    end
end
