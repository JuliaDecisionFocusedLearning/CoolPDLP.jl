using CoolPDLP
using CoolPDLP: Scratch, kkt_errors!
using Random
using SparseArrays
using Test

Random.seed!(0)

milp, _ = CoolPDLP.random_milp_and_sol(10, 20, 0.4)

@testset "$alg parameters" for alg in (PDHG, PDLP)
    algo = alg(Float64, Int, SparseMatrixCSC; check_every = 42, max_kkt_passes = 314)
    str = sprint(show, algo)

    @test occursin("$(nameof(alg)) algorithm", str)
    @test occursin("ConversionParameters", str)
    @test occursin("PreconditioningParameters", str)
    @test occursin("StepSizeParameters", str)
    @test occursin("RestartParameters", str)
    @test occursin("GenericParameters", str)
    @test occursin("TerminationParameters", str)
    # the settings above must survive into the printed configuration
    @test occursin("types=(Float64, Int64, SparseMatrixCSC)", str)
    @test occursin("check_every=42", str)
    @test occursin("max_kkt_passes=314", str)
end

@testset "KKT errors" begin
    sol = PrimalDualSolution(milp)
    err = kkt_errors!(Scratch(sol), sol, milp)
    str = sprint(show, err)
    @test occursin("KKT relative errors", str)
    @test occursin("primal", str) && occursin("dual", str) && occursin("gap", str)
end

@testset "Convergence stats" begin
    _, stats = solve(milp, PDLP(; max_kkt_passes = 200))
    str = sprint(show, stats)
    @test occursin("Convergence stats", str)
    @test occursin(string(stats.termination_status), str)
    @test occursin("KKT passes: $(stats.kkt_passes)", str)
    @test occursin("KKT relative errors", str)
end
