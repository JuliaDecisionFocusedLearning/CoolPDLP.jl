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

err = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol), scratch, sol, milp)
err_p = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol_p), scratch, sol_p, milp_p)

@testset "Correct KKT errors" begin
    @test err.primal ≈ norm(A * x - CoolPDLP.clamp.(A * x, lc, uc))
    @test err.dual ≈ norm(c - At * y - r)
    @test err.gap ≈ abs(dot(c, x) - (p(y, lc, uc) + p(r, lv, uv)))
    @test err.primal_scale ≈ 1 + norm(CoolPDLP.combine.(lc, uc))
    @test err.dual_scale ≈ 1 + norm(c)
    @test err.gap_scale ≈ 1 + abs(dot(c, x)) + abs(p(y, lc, uc) + p(r, lv, uv))
end

@testset "Invariance by preconditioning" begin
    @test err_p ≈ err
end

@testset "Error display" begin
    nbatch = 3
    batch(v) = repeat(v, 1, nbatch)
    milp_batch = MILP(; c = batch(c), lv, uv, A, At, lc, uc)
    sol_batch = PrimalDualSolution(batch(x), batch(y))
    err_batch = CoolPDLP.kkt_errors!(
        CoolPDLP.KKTErrors(sol_batch), CoolPDLP.Scratch(sol_batch), sol_batch, milp_batch
    )

    str, str_batch = sprint(show, err), sprint(show, err_batch)
    @test startswith(str, "KKT relative errors: ")
    @test startswith(str_batch, "KKT relative errors: ")
    # a single value per error without batching, one per instance with it
    @test !occursin('[', str)
    lists = [m.match for m in eachmatch(r"\[[^\]]*\]", str_batch)]
    @test length(lists) == 3
    @test all(list -> count(==(','), list) == nbatch - 1, lists)
end
