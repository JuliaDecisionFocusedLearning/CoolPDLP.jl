using CoolPDLP
using CoolPDLP: termination_status
using GPUArraysCore: @allowscalar
using JLArrays: JLBackend
using KernelAbstractions: CPU
using LinearAlgebra
import MathOptInterface as MOI
using Random
using SparseArrays
using Test

"""
    check_dual_ray(y, milp; atol)

Check that `y` certifies the primal infeasibility of `milp`: its reduced cost `r = -Aᵀy` only
pairs with finite variable bounds, `y` only with finite constraint bounds, and the ray objective
is positive.
"""
function check_dual_ray(y, milp; atol = 1.0e-6)
    (; A, lv, uv, lc, uc) = milp
    y = y / norm(y, Inf)
    r = -(A' * y)
    @test all(@. ifelse(isfinite(lc), true, y <= atol))
    @test all(@. ifelse(isfinite(uc), true, y >= -atol))
    @test all(@. ifelse(isfinite(lv), true, r <= atol))
    @test all(@. ifelse(isfinite(uv), true, r >= -atol))
    term(v, l, u) = (isfinite(l) ? l * max(v, 0) : 0.0) - (isfinite(u) ? u * max(-v, 0) : 0.0)
    @test sum(term.(y, lc, uc)) + sum(term.(r, lv, uv)) > atol
    return nothing
end

"""
    check_primal_ray(d, milp; atol)

Check that `d` certifies the dual infeasibility of `milp`: `A * d` and `d` lie in the recession
cones of the constraint and variable bounds, and `cᵀd < 0`.
"""
function check_primal_ray(d, milp; atol = 1.0e-6)
    (; c, A, lv, uv, lc, uc) = milp
    d = d / norm(d, Inf)
    Ad = A * d
    @test all(@. ifelse(isfinite(lv), d >= -atol, true))
    @test all(@. ifelse(isfinite(uv), d <= atol, true))
    @test all(@. ifelse(isfinite(lc), Ad >= -atol, true))
    @test all(@. ifelse(isfinite(uc), Ad <= atol, true))
    @test dot(c, d) < -atol
    return nothing
end

# x₁ + x₂ ≥ 3 and x₁ + x₂ ≤ 1 with x ≥ 0
infeasible_lp = MILP(;
    c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
    A = sparse([1.0 1.0; 1.0 1.0]), lc = [3.0, -Inf], uc = [Inf, 1.0],
)
# min -x₁ with x₁ - x₂ ≤ 1 and x ≥ 0
unbounded_lp = MILP(;
    c = [-1.0, 0.0], lv = [0.0, 0.0], uv = [Inf, Inf],
    A = sparse([1.0 -1.0]), lc = [-Inf], uc = [1.0],
)
# x₁ - x₂ = 1, x₁ + x₂ = 3, 0 ≤ x ≤ 1: infeasible only through the variable bounds
infeasible_bounds_lp = MILP(;
    c = [1.0, 2.0], lv = [0.0, 0.0], uv = [1.0, 1.0],
    A = sparse([1.0 -1.0; 1.0 1.0]), lc = [1.0, 3.0], uc = [1.0, 3.0],
)
# both primal and dual infeasible: x₁ + x₂ ≥ 1, x₁ + x₂ ≤ -1, min -x₃ with x₃ free
infeasible_unbounded_lp = MILP(;
    c = [0.0, 0.0, -1.0], lv = [-Inf, -Inf, -Inf], uv = [Inf, Inf, Inf],
    A = sparse([1.0 1.0 0.0; 1.0 1.0 0.0]), lc = [1.0, -Inf], uc = [Inf, -1.0],
)

"Make a random LP infeasible with two contradictory copies of a random row."
function random_infeasible_lp(rng, m, n, p)
    milp, _ = CoolPDLP.random_milp_and_sol(rng, m, n, p)
    (; c, lv, uv, A, lc, uc) = milp
    a = sprandn(rng, 1, n, 0.5)
    return MILP(; c, lv, uv, A = [A; a; a], lc = [lc; 1.0; -Inf], uc = [uc; Inf; -1.0])
end

"Make a random LP unbounded with an extra variable which only improves the objective."
function random_unbounded_lp(rng, m, n, p)
    milp, _ = CoolPDLP.random_milp_and_sol(rng, m, n, p)
    (; c, lv, uv, A, lc, uc) = milp
    # x₀ ≥ 0 with cost -1 appears in one more row, `aᵀx - x₀ ≤ 0`, which it can always satisfy
    a = sprandn(rng, 1, n, 0.5)
    return MILP(;
        c = [c; -1.0], lv = [lv; 0.0], uv = [uv; Inf],
        A = [A spzeros(m, 1); a -1.0], lc = [lc; -Inf], uc = [uc; 0.0],
    )
end

algorithms = (PDHG, PDLP, HPDLP, HPRLP)
configs = [(SparseMatrixCSC, CPU()), (GPUSparseMatrixCSR, JLBackend())]

@testset "Hand-made LPs: $alg on $M" for alg in algorithms, (M, backend) in configs
    algo = alg(Float64, Int, M; backend, max_kkt_passes = 10^5, show_progress = false)

    @testset "Primal infeasible" for milp in (infeasible_lp, infeasible_bounds_lp)
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.INFEASIBLE
        @test stats.dual_ray_error <= 1.0e-8
        check_dual_ray(Array(stats.ray.y), milp)
    end

    @testset "Dual infeasible" begin
        sol, stats = solve(unbounded_lp, algo)
        @test termination_status(stats) == MOI.DUAL_INFEASIBLE
        @test stats.primal_ray_error <= 1.0e-8
        check_primal_ray(Array(stats.ray.x), unbounded_lp)
    end

    @testset "Primal and dual infeasible" begin
        sol, stats = solve(infeasible_unbounded_lp, algo)
        # primal infeasibility comes first when both are certified
        @test termination_status(stats) in (MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE)
        if termination_status(stats) == MOI.INFEASIBLE
            check_dual_ray(Array(stats.ray.y), infeasible_unbounded_lp)
        else
            check_primal_ray(Array(stats.ray.x), infeasible_unbounded_lp)
        end
    end

    @testset "Feasible LP is not certified" begin
        milp = MILP(;
            c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
            A = sparse([1.0 1.0]), lc = [1.0], uc = [Inf],
        )
        sol, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
    end
end

@testset "Random infeasible and unbounded LPs: $alg" for alg in algorithms
    algo = alg(; max_kkt_passes = 10^6, show_progress = false)
    @testset "seed $seed" for seed in 1:5
        rng = Xoshiro(seed)
        milp = random_infeasible_lp(rng, 20, 30, 0.3)
        _, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.INFEASIBLE
        check_dual_ray(stats.ray.y, milp)

        milp = random_unbounded_lp(rng, 20, 30, 0.3)
        _, stats = solve(milp, algo)
        # the random part may itself be infeasible, which takes precedence
        @test termination_status(stats) in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE)
        if termination_status(stats) == MOI.DUAL_INFEASIBLE
            check_primal_ray(stats.ray.x, milp)
        else
            check_dual_ray(stats.ray.y, milp)
        end
    end
end

@testset "Crossed bounds" begin
    @testset "$alg" for alg in algorithms
        for milp in (
                MILP(; c = [1.0], lv = [5.0], uv = [2.0], A = sparse([1.0;;]), lc = [0.0], uc = [Inf]),
                MILP(; c = [1.0], lv = [0.0], uv = [2.0], A = sparse([1.0;;]), lc = [1.0], uc = [0.0]),
            )
            _, stats = solve(milp, alg())
            @test termination_status(stats) == MOI.INFEASIBLE
            @test stats.kkt_passes == 0
        end
    end
end

@testset "Crossed bounds in part of a batch: $alg on $M" for alg in algorithms, (M, backend) in configs
    # instance 2 has crossed variable bounds, the others are feasible
    A = sparse([1.0 1.0])
    c = [1.0 1.0 1.0; 1.0 1.0 1.0]
    lv = [0.0 5.0 0.0; 0.0 0.0 0.0]
    uv = [Inf 2.0 Inf; Inf Inf Inf]
    lc, uc = [1.0 1.0 2.0], [Inf Inf Inf]
    milp_batch = MILP(; c, lv, uv, A, lc, uc)
    # with the detection disabled, nothing else could stop instance 2
    algo = alg(Float64, Int, M; backend, infeasibility_tol = nothing, max_kkt_passes = 10^5)
    sol, stats = solve(milp_batch, algo)
    @test termination_status(stats) == MOI.INFEASIBLE
    @allowscalar @test [termination_status(instance(stats, i)) for i in 1:3] ==
        [MOI.OPTIMAL, MOI.INFEASIBLE, MOI.OPTIMAL]
    @test stats.kkt_passes < 10^5
    @allowscalar @test instance(stats, 2).kkt_passes == 0
end

@testset "Type stability" begin
    # on a single instance and on a batch, enabled or disabled
    Random.seed!(0)
    _, milp_batch = random_milp_batch(20, 30, 0.4, 3)
    @testset "$alg, $(nbinstances(milp)) instance(s), tol $tol" for alg in algorithms,
            milp in (infeasible_lp, milp_batch), tol in (1.0e-8, nothing)

        algo = alg(; show_progress = false, infeasibility_tol = tol)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        (; infeas, scratch, stats, sol) = state
        candidates = CoolPDLP.ray_candidates(state)
        @test (@inferred CoolPDLP.infeasibility_check!(stats, infeas, scratch, candidates..., milp, tol, true)) === nothing
        @test (@inferred CoolPDLP.export_certificate!(stats, infeas, milp)) === nothing
        @inferred CoolPDLP.instance(infeas, 1)
        isnothing(tol) && continue
        @inferred CoolPDLP.form_candidate!(infeas.cand, sol, nothing, milp, true)
        @inferred CoolPDLP.primal_ray_error!!(infeas.e, infeas, scratch, infeas.cand.x, milp, tol)
        @inferred CoolPDLP.dual_ray_error!!(infeas.e, infeas, scratch, infeas.cand.y, milp, tol)
        @inferred CoolPDLP.keep_better!!(infeas.ray.x, stats.primal_ray_error, infeas.cand.x, infeas.e, true)
    end
end

@testset "Detection disabled" begin
    @testset "$alg" for alg in algorithms
        algo = alg(; infeasibility_tol = nothing, max_kkt_passes = 1000)
        _, stats = solve(infeasible_lp, algo)
        @test termination_status(stats) == MOI.ITERATION_LIMIT
        _, stats = solve(unbounded_lp, algo)
        @test termination_status(stats) == MOI.ITERATION_LIMIT
    end
end

@testset "Cost of the detection" begin
    # a feasible LP with a negative tolerance, so that only the limit stops it
    milp = MILP(;
        c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 1.0]), lc = [1.0], uc = [Inf],
    )
    kwargs = (; check_every = 10, termination_reltol = -1.0, max_kkt_passes = 100)
    @testset "$alg" for alg in algorithms
        # every check evaluates two candidate rays, at one KKT pass each, counted apart from the
        # passes of the algorithm, so the iteration budget is the same as without detection
        _, stats = solve(milp, alg(; kwargs...))
        @test stats.kkt_passes == 100
        @test stats.ray_passes == 2 * 10

        # disabled, the detection costs nothing: the same KKT passes as without it, i.e. the
        # first multiple of `check_every` past the budget, and no buffer
        algo = alg(; kwargs..., infeasibility_tol = nothing)
        @test isnothing(algo.termination.infeasibility_tol)
        _, stats = solve(milp, algo)
        @test stats.kkt_passes == 100
        @test stats.ray_passes == 0
        @test isnothing(stats.ray)
        @test isinf(stats.primal_ray_error) && isinf(stats.dual_ray_error)

        sol = PrimalDualSolution(milp)
        state = initialize(milp, sol, algo; starting_time = time())
        @test isnothing(state.infeas)
        @test isnothing(state.stats.ray)
        if state isa CoolPDLP.PDHGState
            # PDHG keeps its previous iterate for the infeasibility check only
            @test isnothing(state.sol_last)
        end

        # what the detection allocates, measured on a problem large enough to dominate: at
        # least its two ray buffers and the exported certificate, all gone when disabled
        big, _ = CoolPDLP.random_milp_and_sol(Xoshiro(0), 400, 600, 0.01)
        bigsol = PrimalDualSolution(big)
        init(a) = initialize(big, bigsol, a; starting_time = 0.0)
        enabled, disabled = alg(), alg(; infeasibility_tol = nothing)
        init(enabled), init(disabled)
        buffers = 3 * (nbvar(big) + nbcons(big)) * sizeof(Float64)
        @test @allocated(init(enabled)) - @allocated(init(disabled)) >= buffers
    end
end

@testset "Batch mixing feasible and infeasible instances: $alg on $M" for alg in algorithms, (M, backend) in configs
    # same matrix, three instances: optimal, primal infeasible, dual infeasible
    A = sparse([1.0 1.0; 1.0 -1.0])
    c = [1.0 1.0 -1.0; 1.0 1.0 0.0]
    lc = [1.0 -Inf 1.0; -Inf -Inf -Inf]
    uc = [Inf -1.0 Inf; 1.0 Inf Inf]
    lv, uv = [0.0, 0.0], [Inf, Inf]
    milp_batch = MILP(; c, lv, uv, A, lc, uc)
    milps = [MILP(; c = c[:, i], lv, uv, A, lc = lc[:, i], uc = uc[:, i]) for i in 1:3]

    algo = alg(Float64, Int, M; backend, max_kkt_passes = 10^5, show_progress = false)
    sol, stats = solve(milp_batch, algo)
    # the batch stops once every instance is done, and reports the infeasibility
    @test termination_status(stats) == MOI.INFEASIBLE
    expected = [MOI.OPTIMAL, MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE]
    @allowscalar @test [termination_status(instance(stats, i)) for i in 1:3] == expected
    @test objective_value(Array(sol.x)[:, 1], milps[1]) ≈ 1 atol = 1.0e-3
    check_dual_ray(Array(stats.ray.y)[:, 2], milps[2])
    check_primal_ray(Array(stats.ray.x)[:, 3], milps[3])

    # an instance left running inherits the status of the batch
    algo = alg(Float64, Int, M; backend, max_kkt_passes = 10^5, infeasibility_tol = nothing, show_progress = false)
    _, stats = solve(milp_batch, algo)
    @test termination_status(stats) == MOI.ITERATION_LIMIT
    @allowscalar @test termination_status(instance(stats, 1)) == MOI.OPTIMAL
    @allowscalar @test termination_status(instance(stats, 2)) == MOI.ITERATION_LIMIT
end

@testset "Quadratic programs" begin
    # min ½x₁² - x₂ with x₁ - x₂ ≤ 1 and x ≥ 0: unbounded along d = (0, 1), which has Qd = 0,
    # A d = -1 ≤ 0 and cᵀd = -1 < 0
    unbounded_qp = MILP(;
        c = [0.0, -1.0], Q = sparse([1.0 0.0; 0.0 0.0]), lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 -1.0]), lc = [-Inf], uc = [1.0],
    )
    # min ½x₁² - x₁ with x₁ - x₂ ≤ 1 and x ≥ 0: the linear part alone would be unbounded along
    # (1, 1), but Q curves it up, so the optimum x₁ = 1 is finite
    bounded_qp = MILP(;
        c = [-1.0, 0.0], Q = sparse([1.0 0.0; 0.0 0.0]), lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 -1.0]), lc = [-Inf], uc = [1.0],
    )
    infeasible_qp = MILP(;
        c = [1.0, 1.0], Q = sparse(1.0I, 2, 2), lv = [0.0, 0.0], uv = [Inf, Inf],
        A = sparse([1.0 1.0; 1.0 1.0]), lc = [3.0, -Inf], uc = [Inf, 1.0],
    )
    @testset "$(typeof(algo).parameters[1])" for algo in (
            PDHG(), PDLP(), HPDLP(), HPRQP(),
        )
        sol, stats = solve(unbounded_qp, algo)
        @test termination_status(stats) == MOI.DUAL_INFEASIBLE
        check_primal_ray(stats.ray.x, unbounded_qp)
        d = stats.ray.x / norm(stats.ray.x, Inf)
        @test norm(unbounded_qp.Q * d) < 1.0e-6

        sol, stats = solve(bounded_qp, algo)
        @test termination_status(stats) == MOI.OPTIMAL
        @test objective_value(sol.x, bounded_qp) ≈ -0.5 atol = 1.0e-4

        sol, stats = solve(infeasible_qp, algo)
        @test termination_status(stats) == MOI.INFEASIBLE
        check_dual_ray(stats.ray.y, infeasible_qp)
    end
end
