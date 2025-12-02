using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test

@testset "Involution" begin
    prec = CoolPDLP.Preconditioner(Diagonal(rand(10)), Diagonal(rand(20)))
    @test (inv(prec) * prec).D1 ≈ I
    @test (inv(prec) * prec).D2 ≈ I
    @test (prec * inv(prec)).D1 ≈ I
    @test (prec * inv(prec)).D2 ≈ I
end

@testset "Effect on MILP" begin
    m, n = 10, 20

    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, 0.3)
    lc = rand(m)
    uc = lc + rand(m)
    milp = MILP(; c, lv, uv, A, lc, uc)
    x = randn(n)
    y = randn(m)
    sol = PrimalDualSolution(x, y)

    params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
    prec = CoolPDLP.pdlp_preconditioner(milp, params)

    milp_precond = CoolPDLP.precondition(milp, prec)
    milp_unprecond = CoolPDLP.precondition(milp_precond, inv(prec))
    @test isapprox(milp, milp_unprecond)

    sol_precond = CoolPDLP.precondition(sol, prec)
    sol_unprecond = CoolPDLP.unprecondition(sol_precond, prec)
    @test isapprox(sol, sol_unprecond)
end
