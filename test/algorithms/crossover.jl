using CoolPDLP
using CoolPDLP: termination_status
import MathOptInterface as MOI
using LinearAlgebra
using SparseArrays
using Test

const PDLP_XOVER = (;
    termination_reltol = 1.0e-6,
    max_kkt_passes = 200_000,
    show_progress = false,
)

# min −x₁ − x₂  s.t.  x₁ + x₂ ≤ 1, x ≥ 0  → vertices (1, 0) / (0, 1)
toy_lp() = MILP(;
    c = [-1.0, -1.0],
    lv = [0.0, 0.0],
    uv = [1.0e6, 1.0e6],
    A = sparse([1.0 1.0]),
    lc = [-Inf],
    uc = [1.0],
)

"""Variables strictly inside two finite bounds."""
function n_interior(x, milp; atol = 1.0e-6)
    xv, lv, uv = Array(x), Array(milp.lv), Array(milp.uv)
    return count(eachindex(xv)) do j
        l, u = lv[j], uv[j]
        isfinite(l) && isfinite(u) && xv[j] > l + atol && xv[j] < u - atol
    end
end

@testset "crossover reaches a vertex when OPTIMAL" begin
    milp = toy_lp()
    sol, stats = solve(milp, PDLP(; PDLP_XOVER...))
    x = Array(sol.x)
    @test termination_status(stats) == MOI.OPTIMAL
    @test n_interior(x, milp) <= nbcons(milp)
    @test isapprox(objective_value(x, milp), -1.0; atol = 1.0e-4)
    @test is_feasible(x, milp; cons_tol = 1.0e-4, verbose = false)
end

@testset "no time left keeps the PDLP point" begin
    milp = toy_lp()
    # OPTIMAL is decided before the clock, so a spent limit can still be optimal.
    sol0, stats0 = solve(milp, PDLP(; PDLP_XOVER..., time_limit = 0.0))
    sol1, stats1 = solve(milp, PDLP(; PDLP_XOVER..., time_limit = 1.0))
    @test termination_status(stats0) == MOI.OPTIMAL
    @test termination_status(stats1) == MOI.OPTIMAL
    @test n_interior(sol0.x, milp) > n_interior(sol1.x, milp)
end

@testset "a non-optimal solve skips the crossover" begin
    sol, stats = solve(
        toy_lp(),
        PDLP(;
            termination_reltol = 1.0e-12,
            max_kkt_passes = 1,
            check_every = 1,
            show_progress = false,
        ),
    )
    @test termination_status(stats) != MOI.OPTIMAL
    @test all(isfinite, sol.x)
end

@testset "an infeasible basic solution is not written" begin
    # x₁ + x₂ = 1, 0 ≤ x₂ ≤ 0.2. Basis {x₂} solves to x₂ = 1, outside the box.
    milp = MILP(;
        c = [0.0, 0.0],
        lv = [0.0, 0.0],
        uv = [10.0, 0.2],
        A = sparse([1.0 1.0]),
        lc = [1.0],
        uc = [1.0],
    )
    st = CoolPDLP._lift_boxed(milp, [0.9, 0.1], [0.0])
    st.zero_tol = 1.0e-8
    st.basis .= CoolPDLP._AT_LB
    st.basis[2] = CoolPDLP._BASIC
    st.x[1] = 0.0
    st.x[3] = 1.0
    CoolPDLP._improve_primal!(st)
    x_before = copy(st.x)
    @test CoolPDLP._solve_basic!(st) == false
    @test st.x == x_before
    @test st.x[2] == 0.1
end

@testset "the same seed repeats" begin
    cx = SpiralCrossover{Float64}(; seed = 7)
    s1, _ = solve(toy_lp(), PDLP(; PDLP_XOVER..., crossover = cx))
    s2, _ = solve(toy_lp(), PDLP(; PDLP_XOVER..., crossover = cx))
    @test Array(s1.x) ≈ Array(s2.x)
end

@testset "a wide least-squares ray stays in the kernel" begin
    # b = A c with more columns than rows. The undamped solve makes c − u a
    # nullspace step, which is the primal ray.
    A = sparse([1.0 1.0 0.0; 0.0 1.0 1.0])
    c = [1.0e6, -0.25, -1.0e6]
    b = Vector(A * c)
    u = zeros(3)
    CoolPDLP.ols!(u, A, b)
    @test norm(A * (c - u)) <= 1.0e-8 * (1 + norm(b))
end

@testset "crossover keeps the eltype asked of PDLP" begin
    milp = CoolPDLP.set_eltype(Float32, toy_lp())
    st = CoolPDLP._lift_boxed(milp, fill(Float32(0.5), 2), Float32[1.5])
    @test eltype(st.A) == Float32
    @test eltype(st.x) == Float32
    sol, stats = solve(milp, PDLP(Float32; PDLP_XOVER..., termination_reltol = 1.0f-4))
    @test termination_status(stats) == MOI.OPTIMAL
    @test eltype(sol.x) == Float32
    @test eltype(sol.y) == Float32
end
