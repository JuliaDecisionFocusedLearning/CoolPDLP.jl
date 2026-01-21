@testset "Termination check not skipped" begin
    c = [1.0, 1.0]
    lc, A, uc = [1.0], sparse([1.0 1.0]), [Inf]
    lv, uv = [0.0, 0.0], [Inf, Inf]

    milp = CoolPDLP.MILP(; c, lv, uv, A, lc, uc)
    algo = CoolPDLP.PDLP()
    sol, stats = CoolPDLP.solve(milp, algo)
    @test stats.termination_status == CoolPDLP.OPTIMAL
end
