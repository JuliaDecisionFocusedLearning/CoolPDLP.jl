using CoolPDLP
using CoolPDLP: BatchedGPUSparseMatrixCSR, instance
using LinearAlgebra
using SparseArrays
using Random: Xoshiro
using Test

include("../fixtures.jl")

@testset "Composition" begin
    A = sprand(10, 20, 0.4)
    cons = CoolPDLP.ConstraintMatrix(A, sparse(transpose(A)))
    prec_in = CoolPDLP.Preconditioner(Diagonal(rand(10)), Diagonal(rand(20)))
    prec_out = CoolPDLP.Preconditioner(Diagonal(rand(10)), Diagonal(rand(20)))
    prec = prec_out * prec_in
    cons_p1 = CoolPDLP.precondition(cons, prec)
    cons_p2 = CoolPDLP.precondition(CoolPDLP.precondition(cons, prec_in), prec_out)
    @test cons_p1.A ≈ cons_p2.A
    @test cons_p1.At ≈ cons_p2.At
end

@testset "Involution" begin
    prec = CoolPDLP.Preconditioner(Diagonal(rand(10)), Diagonal(rand(20)))
    @test (inv(prec) * prec).D1 ≈ I
    @test (inv(prec) * prec).D2 ≈ I
    @test (prec * inv(prec)).D1 ≈ I
    @test (prec * inv(prec)).D2 ≈ I
end

@testset "Preconditioner types" begin
    rng = Xoshiro(42)
    A = sprand(rng, 10, 20, 0.4)
    cons = CoolPDLP.ConstraintMatrix(A, sparse(transpose(A)))
    @testset "Identity" begin
        id_prec = CoolPDLP.identity_preconditioner(cons)
        @test id_prec.D1 == I
        @test id_prec.D2 == I
    end
    @testset "Ruiz" begin
        ruiz_prec = CoolPDLP.ruiz_preconditioner(cons; iterations = 10000)
        cons_p = CoolPDLP.precondition(cons, ruiz_prec)
        @test all(≈(1; rtol = 1.0e-2), map(col -> norm(col, Inf), eachcol(cons_p.A)))
        @test all(≈(1; rtol = 1.0e-2), map(col -> norm(col, Inf), eachcol(cons_p.At)))
    end
end

@testset "Effect on MILP" begin
    m, n, p = 10, 20, 0.4

    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, p)
    lc = rand(m)
    uc = lc + rand(m)
    milp = MILP(; c, lv, uv, A, lc, uc)
    x = randn(n)
    y = randn(m)
    sol = PrimalDualSolution(x, y)

    params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
    prec = CoolPDLP.pdlp_preconditioner(milp, params)

    milp_p = CoolPDLP.precondition(milp, prec)
    milp_unp = CoolPDLP.precondition(milp_p, inv(prec))
    @test isapprox(milp, milp_unp)
    @test !isapprox(milp, milp_p)

    sol_p = CoolPDLP.precondition(sol, prec)
    sol_unp = CoolPDLP.unprecondition(sol_p, prec)
    @test isapprox(sol, sol_unp)
    @test !isapprox(sol, sol_p)

    @test objective_value(sol.x, milp) ≈ objective_value(sol_p.x, milp_p)
    @test dot(sol.y, milp.A, sol.x) ≈ dot(sol_p.y, milp_p.A, sol_p.x)
    @test CoolPDLP.clamp.(sol.x, milp.lv, milp.uv) ≈ prec.D2 * CoolPDLP.clamp.(sol_p.x, milp_p.lv, milp_p.uv)
    @test CoolPDLP.clamp.(milp.A * sol.x, milp.lc, milp.uc) ≈ prec.D1 \ CoolPDLP.clamp.(milp_p.A * sol_p.x, milp_p.lc, milp_p.uc)
end

@testset "Effect on a batch of MILPs" begin
    rng = Xoshiro(0)
    m, n, nbatch = 10, 20, 3
    pattern = sprand(rng, m, n, 0.4)
    As = [same_pattern(pattern, rng) for _ in 1:nbatch]
    c, lv = rand(rng, n), rand(rng, n)
    uv, lc = lv + rand(rng, n), rand(rng, m)
    uc = lc + rand(rng, m)
    milps = map(A -> MILP(; c, lv, uv, A, lc, uc), As)
    milp = MILP(; c, lv, uv, A = BatchedGPUSparseMatrixCSR(As), lc, uc)
    sol = PrimalDualSolution(randn(rng, n, nbatch), randn(rng, m, nbatch))

    params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
    prec = CoolPDLP.pdlp_preconditioner(milp, params)
    milp_p = CoolPDLP.precondition(milp, prec)
    # unpreconditioning gives every instance its own copy of the shared fields back
    milp_unp = CoolPDLP.precondition(milp_p, inv(prec))

    @test size(prec.D1) == (m, m, nbatch)
    @test size(prec.D2) == (n, n, nbatch)

    @testset "instance $i" for i in 1:nbatch
        prec_i = CoolPDLP.pdlp_preconditioner(milps[i], params)
        @test CoolPDLP.instance(prec.D1, i) ≈ prec_i.D1
        @test CoolPDLP.instance(prec.D2, i) ≈ prec_i.D2
        @test same_instance(instance(milp_p, i), CoolPDLP.precondition(milps[i], prec_i))
        @test same_instance(instance(milp_unp, i), milps[i])
    end

    sol_p = CoolPDLP.precondition(sol, prec)
    @test isapprox(sol, CoolPDLP.unprecondition(sol_p, prec))
    @test !isapprox(sol, sol_p)
end
