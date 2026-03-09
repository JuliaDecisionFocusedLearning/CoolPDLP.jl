using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test

function p(y, l, u)
    y⁺ = CoolPDLP.positive_part.(y)
    y⁻ = CoolPDLP.negative_part.(y)
    u_noinf = CoolPDLP.safe.(u)
    l_noinf = CoolPDLP.safe.(l)
    return dot(y⁺, u_noinf) - dot(y⁻, l_noinf)
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
    @test err.gap ≈ abs(dot(c, x) + p(-y, lc, uc) + p(-r, lv, uv))
    @test err.primal_scale ≈ 1 + norm(CoolPDLP.combine.(lc, uc))
    @test err.dual_scale ≈ 1 + norm(c)
    @test err.gap_scale ≈ 1 + abs(dot(c, x)) + abs(p(-y, lc, uc) + p(-r, lv, uv))
end

@testset "Invariance by preconditioning" begin
    @test err_p ≈ err
end

n, m = 20, 10
c = randn(n)
lv = zeros(n)
uv = fill(Inf, n)
A = sprand(m, n, 0.4)
At = sparse(A')
lc = randn(m)
uc = lc + rand(m)
H = sprand(n, n, 0.3)
Q = Matrix(H' * H)

qp = QuadraticProgram(; c, lv, uv, A, Q = sparse(Q), lc, uc)
x = abs.(randn(n))
y = randn(m)
sol_qp = PrimalDualSolution(x, y)
scratch_qp = CoolPDLP.Scratch(sol_qp)

err_qp = CoolPDLP.kkt_errors!(scratch_qp, sol_qp, qp)

g = c + Q * x
r_qp = CoolPDLP.proj_multiplier.(g - At * y, lv, uv)
half_xQx = dot(x, Q * x) / 2
cx = dot(g, x)  # = cᵀx + xᵀQx
pc_sum = p(-y, lc, uc)
pv_sum = p(-r_qp, lv, uv)

@testset "Correct QP KKT errors" begin
    @test err_qp.primal ≈ norm(A * x - CoolPDLP.proj_box.(A * x, lc, uc))
    @test err_qp.dual ≈ norm(g - At * y - r_qp)
    @test err_qp.gap ≈ abs(cx + pc_sum + pv_sum)
    @test err_qp.dual_scale ≈ 1 + norm(g)
    @test err_qp.gap_scale ≈ 1 + abs(cx - half_xQx) + abs(pc_sum + pv_sum + half_xQx)
end

@testset "QP KKT invariance by preconditioning" begin
    params_qp = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1.0, ruiz_iter = 5)
    prec_qp = CoolPDLP.pdlp_preconditioner(qp, params_qp)
    qp_p = CoolPDLP.precondition(qp, prec_qp)
    sol_qp_p = CoolPDLP.precondition(sol_qp, prec_qp)
    scratch_qp_p = CoolPDLP.Scratch(sol_qp_p)
    err_qp_p = CoolPDLP.kkt_errors!(scratch_qp_p, sol_qp_p, qp_p)
    @test err_qp_p ≈ err_qp
end
