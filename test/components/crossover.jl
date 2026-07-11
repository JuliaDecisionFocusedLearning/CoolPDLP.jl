using CoolPDLP
using LinearAlgebra
using Random
using SparseArrays
using Test

const _PDLP_TEST_KW = (;
    termination_reltol = 1.0e-4,
    max_kkt_passes = 200_000,
    show_progress = false,
    check_every = 10,
)

function _kkt_err(primal, dual, gap)
    return CoolPDLP.KKTErrors(;
        primal,
        primal_scale = 1.0,
        dual,
        dual_scale = 1.0,
        gap,
        gap_scale = 1.0,
    )
end

const _TOY_LP = MILP(;
    c = [1.0, 2.0],
    lv = [0.0, 0.0],
    uv = [Inf, Inf],
    A = sparse([1.0 1.0]),
    lc = [1.0],
    uc = [1.0],
)

const _BOX_LP = MILP(;
    c = [1.0, 2.0],
    lv = [0.0, 0.0],
    uv = [1.0, 1.0],
    A = sparse([1.0 1.0]),
    lc = [1.0],
    uc = [1.0],
)

@testset "crossover_threshold!" begin
    x = [0.5, 0.999999, 2.0]
    params = CoolPDLP.CrossoverParameters(; threshold = 1.0e-5)
    CoolPDLP.crossover_threshold!(x, [0.0, 0.0, -Inf], [1.0, 1.0, Inf], params)
    @test x == [0.5, 1.0, 2.0]

    x = [1.0 - 1.0e-7, 1.0e-7]
    CoolPDLP.crossover_threshold!(x, _BOX_LP, params)
    @test x ≈ [1.0, 0.0]
end

@testset "crossover_effective_bounds" begin
    x = [1.0 - 1.0e-7, 0.0]
    _, uv = CoolPDLP.crossover_effective_bounds(_TOY_LP, x)
    @test isapprox(uv[1], 1.0; atol = 1.0e-12)
    @test isinf(uv[2])

    milp_neg = MILP(;
        c = [1.0, 1.0],
        lv = [-Inf, 0.0],
        uv = [Inf, 1.0],
        A = sparse([-1.0 1.0]),
        lc = [1.0],
        uc = [1.0],
    )
    lv_neg, _ = CoolPDLP.crossover_effective_bounds(milp_neg, [0.5, 1.0])
    @test isapprox(lv_neg[1], 0.0; atol = 1.0e-12)

    milp_min = MILP(;
        c = [1.0, 2.0],
        lv = [0.0, 0.0],
        uv = [0.5, Inf],
        A = sparse([1.0 1.0]),
        lc = [1.0],
        uc = [1.0],
    )
    _, uv_min = CoolPDLP.crossover_effective_bounds(milp_min, x)
    @test isapprox(uv_min[1], 0.5; atol = 1.0e-12)

    milp_multi = MILP(;
        c = ones(3),
        lv = zeros(3),
        uv = fill(Inf, 3),
        A = sparse([1.0 1.0 1.0]),
        lc = [1.0],
        uc = [1.0],
    )
    lv_m, uv_m = CoolPDLP.crossover_effective_bounds(milp_multi, [0.2, 0.3, 0.5])
    @test lv_m == milp_multi.lv && uv_m == milp_multi.uv

    using Adapt
    using JLArrays: JLBackend
    milp_gpu = adapt(JLBackend(), CoolPDLP.set_matrix_type(CoolPDLP.GPUSparseMatrixCSR, _TOY_LP))
    x_gpu = adapt(JLBackend(), x)
    _, uv_gpu = CoolPDLP.crossover_effective_bounds(milp_gpu, x_gpu)
    @test isapprox(Array(uv_gpu)[1], 1.0; atol = 1.0e-12)
end

@testset "crossover_kkt_acceptable and rollback" begin
    err_lo = _kkt_err(1.0e-5, 1.0e-5, 1.0e-5)
    err_hi = _kkt_err(2.0e-4, 2.0e-4, 2.0e-4)
    err_mid = _kkt_err(1.05e-5, 1.05e-5, 1.05e-5)
    tol = 1.0e-4
    strict = CoolPDLP.CrossoverParameters(; rollback_on_kkt_regression = true, kkt_rtol = 0.0)
    slack = CoolPDLP.CrossoverParameters(; rollback_on_kkt_regression = true, kkt_rtol = 0.1)

    @test CoolPDLP.crossover_kkt_acceptable(err_lo, err_lo, tol, strict)
    @test !CoolPDLP.crossover_kkt_acceptable(err_lo, err_hi, tol, strict)
    @test CoolPDLP.crossover_kkt_acceptable(err_lo, err_mid, tol, slack)
    @test CoolPDLP.crossover_kkt_acceptable(
        err_lo,
        err_hi,
        tol,
        CoolPDLP.CrossoverParameters(; rollback_on_kkt_regression = false),
    )

    milp = MILP(;
        c = [0.0, 1.0],
        lv = [0.0, 0.0],
        uv = [1.0, 1.0],
        A = sparse([1.0 1.0]),
        lc = [0.6],
        uc = [0.6],
    )
    algo_kw = (;
        crossover = true,
        crossover_threshold = 0.5,
        crossover_use_effective_bounds = false,
        _PDLP_TEST_KW...,
    )
    Random.seed!(123)
    _, stats_rb = solve(milp, PDLP(; crossover_rollback_on_kkt_regression = true, algo_kw...))
    Random.seed!(123)
    _, stats_no = solve(
        milp,
        PDLP(; crossover_rollback_on_kkt_regression = false, algo_kw...),
    )
    @test stats_rb.crossover_rolled_back && !stats_rb.crossover_applied
    @test stats_no.crossover_applied && !stats_no.crossover_rolled_back
    @test CoolPDLP.relative(stats_rb.err) <= tol
end

@testset "crossover on solve" begin
    Random.seed!(42)
    sol_off, stats_off = solve(_TOY_LP, PDLP(; crossover = false, _PDLP_TEST_KW...))
    Random.seed!(42)
    sol_on, stats_on = solve(_TOY_LP, PDLP(; crossover = true, _PDLP_TEST_KW...))
    x_off, x_on = Array(sol_off.x), Array(sol_on.x)

    @test stats_on.termination_status == CoolPDLP.OPTIMAL
    @test stats_on.crossover_applied
    @test stats_on.crossover_n_snapped >= 1
    @test !stats_on.crossover_rolled_back
    @test x_on[1] ≈ 1.0 atol = 1.0e-12
    @test x_on[1] > x_off[1]
    @test CoolPDLP.is_feasible(x_on, _TOY_LP; cons_tol = 1.0e-5, verbose = false)

    sol_box, stats_box = solve(_BOX_LP, PDLP(; crossover = true, _PDLP_TEST_KW...))
    @test stats_box.termination_status == CoolPDLP.OPTIMAL
    @test CoolPDLP.fraction_at_bounds(Array(sol_box.x), _BOX_LP) == 1.0
    @test !stats_box.crossover_rolled_back

    _, stats_disabled = solve(_BOX_LP, PDLP(; crossover = false, _PDLP_TEST_KW...))
    @test !stats_disabled.crossover_applied
    @test stats_disabled.crossover_n_snapped == 0
end

@testset "ConvergenceStats show" begin
    stats = CoolPDLP.ConvergenceStats(Float64)
    @test occursin("crossover not applied", sprint(show, stats))
    stats.crossover_applied = true
    stats.crossover_n_snapped = 3
    @test occursin("crossover applied (3 coords)", sprint(show, stats))
    stats.crossover_applied = false
    stats.crossover_rolled_back = true
    @test occursin("crossover rolled back", sprint(show, stats))
end

@testset "crossover_n_changed" begin
    @test CoolPDLP.crossover_n_changed([1.0, 2.0], [1.0, 2.0]) == 0
    @test CoolPDLP.crossover_n_changed([1.0, 0.0], [1.0, 2.0]) == 1

    using Adapt
    using JLArrays: JLBackend
    x_before = adapt(JLBackend(), [1.0, 2.0])
    x_after = copy(x_before)
    x_after .= [1.0, 0.0]
    @test CoolPDLP.crossover_n_changed(x_after, x_before) == 1
end
