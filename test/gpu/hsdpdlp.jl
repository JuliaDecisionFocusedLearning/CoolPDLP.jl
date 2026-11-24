using CoolPDLP
using CoolPDLP: GPUSparseMatrixCSR, initialize, termination_status
using JLArrays: JLArrays, JLArray, JLBackend
using MathOptBenchmarkInstances: Netlib, read_instance
using Test

"""
    test_hsdpdlp_device(M, backend, ArrayType)

Solve an unbatched HSDPDLP on a device without scalar indexing, and compare with the CPU.
"""
function test_hsdpdlp_device(M, backend, ArrayType)
    milp = MILP(read_instance(Netlib, "afiro")[1])
    kw = (; termination_reltol = 1.0e-6, show_progress = false)
    sol_cpu, stats_cpu = solve(milp, HSDPDLP(Float64, Int, M; kw...))
    algo = HSDPDLP(Float64, Int, M; backend, kw...)

    @testset "Scalars stay on the device" begin
        # unbatched, every reduction of a step would read a host number back: the solve runs on a
        # batch of one, whose scalars are device vectors
        milp_d, sol_d = preprocess(milp, PrimalDualSolution(milp), algo)
        @test isbatched(milp_d)
        @test size(sol_d.x) == (nbvar(milp), 1)
        state = initialize(milp_d, sol_d, algo; starting_time = 0.0)
        @test state.ws.t isa ArrayType
        @test state.ws.g isa ArrayType
        @test state.z.τ isa ArrayType
        @test state.halpern.fpe isa ArrayType
        @test state.step_sizes.ω isa ArrayType
    end

    @testset "Same result as on the CPU" begin
        sol, stats = solve(milp, algo)
        @test sol.x isa ArrayType
        @test ndims(sol.x) == 1
        @test stats.err.primal isa Number
        @test termination_status(stats) == termination_status(stats_cpu)
        @test Array(sol.x) ≈ sol_cpu.x rtol = 1.0e-5 atol = 1.0e-6
        @test Array(sol.y) ≈ sol_cpu.y rtol = 1.0e-5 atol = 1.0e-6
    end
    return nothing
end
