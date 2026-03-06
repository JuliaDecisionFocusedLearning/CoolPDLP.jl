using CoolPDLP
using LinearAlgebra
using SparseArrays
using Random: Xoshiro
using Test

function _isapprox(m1::LinearProgram, m2::LinearProgram; kwargs...)
    return (
        isapprox(m1.c, m2.c; kwargs...) &&
            isapprox(m1.lv, m2.lv; kwargs...) &&
            isapprox(m1.uv, m2.uv; kwargs...) &&
            isapprox(m1.A, m2.A; kwargs...) &&
            isapprox(m1.At, m2.At; kwargs...) &&
            isapprox(m1.lc, m2.lc; kwargs...) &&
            isapprox(m1.uc, m2.uc; kwargs...) &&
            isapprox(m1.D1, m2.D1; kwargs...) &&
            isapprox(m1.D2, m2.D2; kwargs...)
    )
end

function _isapprox(m1::QuadraticProgram, m2::QuadraticProgram; kwargs...)
    return (
        isapprox(m1.c, m2.c; kwargs...) &&
            isapprox(m1.lv, m2.lv; kwargs...) &&
            isapprox(m1.uv, m2.uv; kwargs...) &&
            isapprox(m1.A, m2.A; kwargs...) &&
            isapprox(m1.At, m2.At; kwargs...) &&
            isapprox(m1.Q, m2.Q; kwargs...) &&
            isapprox(m1.lc, m2.lc; kwargs...) &&
            isapprox(m1.uc, m2.uc; kwargs...) &&
            isapprox(m1.D1, m2.D1; kwargs...) &&
            isapprox(m1.D2, m2.D2; kwargs...)
    )
end

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
        milp = LinearProgram(; c = zeros(20), lv = zeros(20), uv = ones(20), A, lc = zeros(10), uc = ones(10))
        ruiz_prec, cons_p = CoolPDLP.ruiz_preconditioner(milp; iterations = 10000)
        @test all(≈(1; rtol = 1.0e-2), map(col -> norm(col, Inf), eachcol(cons_p.A)))
        @test all(≈(1; rtol = 1.0e-2), map(col -> norm(col, Inf), eachcol(cons_p.At)))
    end
end

@testset "Effect on LinearProgram" begin
    m, n, p = 10, 20, 0.4

    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, p)
    lc = rand(m)
    uc = lc + rand(m)
    milp = LinearProgram(; c, lv, uv, A, lc, uc)
    x = randn(n)
    y = randn(m)
    sol = PrimalDualSolution(x, y)

    params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
    prec = CoolPDLP.pdlp_preconditioner(milp, params)

    milp_p = CoolPDLP.precondition(milp, prec)
    milp_unp = CoolPDLP.precondition(milp_p, inv(prec))
    @test _isapprox(milp, milp_unp)
    @test !_isapprox(milp, milp_p)

    sol_p = CoolPDLP.precondition(sol, prec)
    sol_unp = CoolPDLP.unprecondition(sol_p, prec)
    @test isapprox(sol, sol_unp)
    @test !isapprox(sol, sol_p)

    @test objective_value(sol.x, milp) ≈ objective_value(sol_p.x, milp_p)
    @test dot(sol.y, milp.A, sol.x) ≈ dot(sol_p.y, milp_p.A, sol_p.x)
    @test CoolPDLP.proj_box.(sol.x, milp.lv, milp.uv) ≈ prec.D2 * CoolPDLP.proj_box.(sol_p.x, milp_p.lv, milp_p.uv)
    @test CoolPDLP.proj_box.(milp.A * sol.x, milp.lc, milp.uc) ≈ prec.D1 \ CoolPDLP.proj_box.(milp_p.A * sol_p.x, milp_p.lc, milp_p.uc)
end

@testset "Effect on QuadraticProgram" begin
    m, n, p = 10, 20, 0.4

    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, p)
    Q = let H = sprand(n, n, p)
        H' * H
    end
    lc = rand(m)
    uc = lc + rand(m)
    qp = QuadraticProgram(; c, lv, uv, A, Q, lc, uc)
    x = randn(n)
    y = randn(m)
    sol = PrimalDualSolution(x, y)

    params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
    prec = CoolPDLP.pdlp_preconditioner(qp, params)

    qp_p = CoolPDLP.precondition(qp, prec)
    qp_unp = CoolPDLP.precondition(qp_p, inv(prec))
    @test _isapprox(qp, qp_unp)
    @test !_isapprox(qp, qp_p)
    @test !(qp.Q ≈ qp_p.Q)

    sol_p = CoolPDLP.precondition(sol, prec)
    sol_unp = CoolPDLP.unprecondition(sol_p, prec)
    @test isapprox(sol, sol_unp)
    @test !isapprox(sol, sol_p)

    @test objective_value(sol.x, qp) ≈ objective_value(sol_p.x, qp_p)
    @test dot(sol.y, qp.A, sol.x) ≈ dot(sol_p.y, qp_p.A, sol_p.x)
    @test CoolPDLP.proj_box.(sol.x, qp.lv, qp.uv) ≈ prec.D2 * CoolPDLP.proj_box.(sol_p.x, qp_p.lv, qp_p.uv)
    @test CoolPDLP.proj_box.(qp.A * sol.x, qp.lc, qp.uc) ≈ prec.D1 \ CoolPDLP.proj_box.(qp_p.A * sol_p.x, qp_p.lc, qp_p.uc)
end
