using CoolPDLP
using CoolPDLP: instance, termination_status
using HiGHS: HiGHS
using JuMP: JuMP, MOI
using LinearAlgebra
using MathOptBenchmarkInstances
using SparseArrays
using Random: Xoshiro
using Test

"Spread of the nonzero magnitudes of `A`, as the ratio of the largest to the smallest."
nz_spread(A) = (v = abs.(nonzeros(A)); maximum(v) / minimum(v))
"Curtis-Reid objective: sum of squared logs of the nonzero magnitudes of `A`."
log_sq(A) = sum(abs2 ∘ log ∘ abs, nonzeros(A))

@testset "Matrix scalings" begin
    rng = Xoshiro(0)
    A = sprand(rng, 30, 50, 0.2)
    A = Diagonal(exp.(4 .* randn(rng, 30))) * A * Diagonal(exp.(4 .* randn(rng, 50)))
    cons = CoolPDLP.ConstraintMatrix(A, sparse(transpose(A)))
    @testset "Geometric mean" begin
        prec = CoolPDLP.geometric_mean_preconditioner(cons; iterations = 12)
        A_p = CoolPDLP.precondition(cons, prec).A
        @test all(>(0), prec.D1.diag) && all(>(0), prec.D2.diag)
        @test nz_spread(A_p) < nz_spread(A) / 100
        # the transposed matrix is scaled consistently
        @test CoolPDLP.precondition(cons, prec).At ≈ sparse(transpose(A_p))
    end
    @testset "Curtis-Reid" begin
        prec = CoolPDLP.curtis_reid_preconditioner(cons; iterations = 20)
        A_p = CoolPDLP.precondition(cons, prec).A
        @test all(>(0), prec.D1.diag) && all(>(0), prec.D2.diag)
        @test log_sq(A_p) < log_sq(A) / 10
        # more iterations never increase the least-squares objective (block coordinate descent)
        prec_more = CoolPDLP.curtis_reid_preconditioner(cons; iterations = 40)
        @test log_sq(CoolPDLP.precondition(cons, prec_more).A) <= log_sq(A_p) * (1 + 1.0e-10)
    end
    @testset "Curtis-Reid skips stored zeros, in $T" for T in (Float64, Float32)
        B = sparse(T[2 4; 0 8])
        # the same matrix with an explicitly stored zero at (2, 1)
        Bz = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2], T[2, 0, 4, 8])
        @test nnz(Bz) == 4
        cons_z = CoolPDLP.ConstraintMatrix(Bz, sparse(transpose(Bz)))
        cons_b = CoolPDLP.ConstraintMatrix(dropzeros(B), sparse(transpose(dropzeros(B))))
        prec_z = CoolPDLP.curtis_reid_preconditioner(cons_z; iterations = 20)
        prec_b = CoolPDLP.curtis_reid_preconditioner(cons_b; iterations = 20)
        @test eltype(prec_z.D1.diag) == T
        @test all(isfinite, prec_z.D1.diag) && all(isfinite, prec_z.D2.diag)
        @test prec_z.D1.diag ≈ prec_b.D1.diag
        @test prec_z.D2.diag ≈ prec_b.D2.diag
    end
    @testset "Disabled by default" begin
        milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)
        params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1.0, ruiz_iter = 10)
        params_off = CoolPDLP.PreconditioningParameters(;
            chambolle_pock_alpha = 1.0, ruiz_iter = 10,
            geometric_mean_iter = 0, curtis_reid_iter = 0, bound_objective_rescaling = false,
        )
        @test CoolPDLP.pdlp_preconditioner(milp, params).D1 == CoolPDLP.pdlp_preconditioner(milp, params_off).D1
        @test !PDLP().preconditioning.bound_objective_rescaling
        @test PDLP().preconditioning.geometric_mean_iter == 0
        @test CoolPDLP.Algorithm{:PDLP}().preconditioning.curtis_reid_iter == 0
        # `PDLP()` starts with 20 Curtis-Reid rounds, and one keyword turns them off
        @test PDLP().preconditioning.curtis_reid_iter == 20
        @test PDLP(; curtis_reid_iter = 0).preconditioning.curtis_reid_iter == 0
    end
end

"Precondition with every stage on, then rescale bounds and objective, like `preprocess`."
function full_scaling(milp, sol)
    params = CoolPDLP.PreconditioningParameters(;
        chambolle_pock_alpha = 1.0, ruiz_iter = 10,
        geometric_mean_iter = 12, curtis_reid_iter = 20, bound_objective_rescaling = true,
    )
    prec = CoolPDLP.pdlp_preconditioner(milp, params)
    milp_p, sol_p = CoolPDLP.precondition(milp, prec), CoolPDLP.precondition(sol, prec)
    β, γ = CoolPDLP.bound_objective_factors(milp_p)
    return CoolPDLP.rescale_bounds_objective(milp_p, β, γ),
        CoolPDLP.rescale_bounds_objective(sol_p, β, γ), β, γ
end

kkt(sol, milp) = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol), CoolPDLP.Scratch(sol), sol, milp)

@testset "Bound and objective rescaling round trip" begin
    milp, sol = CoolPDLP.random_milp_and_sol(40, 60, 0.3)
    milp_s, sol_s, β, γ = full_scaling(milp, sol)
    @test 0 < β < 1 && 0 < γ < 1
    @test milp_s.bound_scale == β
    @test milp_s.obj_scale == γ
    @test CoolPDLP.unscale(sol_s, milp_s).x ≈ sol.x
    @test CoolPDLP.unscale(sol_s, milp_s).y ≈ sol.y
    # the rescaled bounds and objective have norm below one
    @test norm(CoolPDLP.combine.(milp_s.lc, milp_s.uc)) < 1
    @test norm(milp_s.c) < 1
    # errors, relative scales included, are those of the original problem
    @test kkt(sol_s, milp_s) ≈ kkt(sol, milp)
end

@testset "Batched rescaling is per instance" begin
    milp, sol = CoolPDLP.random_milp_and_sol(40, 60, 0.3)
    (; c, c0, lv, uv, A, At, lc, uc) = milp
    factors = [1.0, 10.0, 1000.0]
    milp_batch = MILP(;
        c = c .* factors', c0, lv, uv, A, At,
        lc = lc .* reverse(factors)', uc = uc .* reverse(factors)',
    )
    sol_batch = PrimalDualSolution(repeat(sol.x, 1, 3), repeat(sol.y, 1, 3))
    milp_s, sol_s, β, γ = full_scaling(milp_batch, sol_batch)
    @test β isa Vector && length(β) == 3 && allunique(β)
    @test γ isa Vector && length(γ) == 3 && allunique(γ)
    @test CoolPDLP.unscale(sol_s, milp_s).x ≈ sol_batch.x
    @test CoolPDLP.unscale(sol_s, milp_s).y ≈ sol_batch.y
    err_batch = kkt(sol_s, milp_s)
    for i in 1:3
        milp_i = instance(milp_batch, i)
        milp_si, _, βi, γi = full_scaling(milp_i, sol)
        @test βi ≈ β[i]
        @test γi ≈ γ[i]
        @test instance(milp_s, i).bound_scale ≈ βi
        @test isapprox(instance(milp_s, i), milp_si)
        @test instance(err_batch, i) ≈ kkt(sol, milp_i)
    end
end

@testset "Solve with the new stages" begin
    for name in ["afiro", "sc50a"]
        qps, path = read_instance(Netlib, name)
        milp = MILP(qps; dataset = Netlib, path)
        jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
        JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
        JuMP.set_silent(jump_model)
        JuMP.optimize!(jump_model)
        obj_highs = JuMP.objective_value(jump_model)
        sol_ref, _ = solve(milp, PDLP(; termination_reltol = 1.0e-6))
        @testset "$name $kw" for kw in [
                (; geometric_mean_iter = 12),
                (; curtis_reid_iter = 20),
                (; bound_objective_rescaling = true),
                (; geometric_mean_iter = 12, bound_objective_rescaling = true),
            ]
            sol, stats = solve(milp, PDLP(; termination_reltol = 1.0e-6, kw...))
            @test termination_status(stats) == MOI.OPTIMAL
            @test objective_value(sol.x, milp) ≈ obj_highs rtol = 1.0e-4
            @test objective_value(sol.x, milp) ≈ objective_value(sol_ref.x, milp) rtol = 1.0e-4
            @test is_feasible(sol.x, milp; cons_tol = 1.0e-3)
            # the reported errors are those of the returned (original-scale) solution
            @test CoolPDLP.relative(kkt(sol, milp)) ≈ CoolPDLP.relative(stats.err) rtol = 1.0e-6
        end
    end
end

@testset "Batched solve with rescaling" begin
    milp = MILP(read_instance(Netlib, "afiro")[1])
    (; c, c0, lv, uv, A, At, lc, uc) = milp
    factors = [1.0, 100.0]
    milp_batch = MILP(; c = c .* factors', c0, lv, uv, A, At, lc, uc)
    algo = PDLP(; termination_reltol = 1.0e-6, bound_objective_rescaling = true)
    sol_batch, stats = solve(milp_batch, algo)
    for i in 1:2
        @test termination_status(instance(stats, i)) == MOI.OPTIMAL
        sol_i, _ = solve(instance(milp_batch, i), algo)
        @test objective_value(sol_batch.x[:, i], instance(milp_batch, i)) ≈
            objective_value(sol_i.x, instance(milp_batch, i)) rtol = 1.0e-4
    end
end
