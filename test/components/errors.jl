using CoolPDLP
using LinearAlgebra
using Test

function p(y, l, u)
    y⁺ = CoolPDLP.positive_part.(y)
    y⁻ = CoolPDLP.negative_part.(y)
    u_noinf = CoolPDLP.safe.(u)
    l_noinf = CoolPDLP.safe.(l)
    return dot(y⁺, l_noinf) - dot(y⁻, u_noinf)
end

milp, sol = CoolPDLP.random_milp_and_sol(100, 200, 0.4)
(; c, lv, uv, A, At, lc, uc, D1, D2) = milp
(; x, y) = sol
r = CoolPDLP.proj_multiplier.(c - At * y, lv, uv)
scratch = CoolPDLP.Scratch(sol)

params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1.0, ruiz_iter = 10)
prec = CoolPDLP.pdlp_preconditioner(milp, params)
milp_p = CoolPDLP.precondition(milp, prec)
sol_p = CoolPDLP.precondition(sol, prec)

err = CoolPDLP.kkt_errors!(scratch, sol, milp)
err_p = CoolPDLP.kkt_errors!(scratch, sol_p, milp_p)

@testset "Correct KKT errors" begin
    @test err.primal ≈ norm(A * x - CoolPDLP.proj_box.(A * x, lc, uc))
    @test err.dual ≈ norm(c - At * y - r)
    @test err.gap ≈ abs(dot(c, x) - (p(y, lc, uc) + p(r, lv, uv)))
    @test err.primal_scale ≈ 1 + norm(CoolPDLP.combine.(lc, uc))
    @test err.dual_scale ≈ 1 + norm(c)
    @test err.gap_scale ≈ 1 + abs(dot(c, x)) + abs(p(y, lc, uc) + p(r, lv, uv))
end

@testset "Invariance by preconditioning" begin
    @test err_p ≈ err
end
