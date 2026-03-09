using CoolPDLP
using JLArrays
using JuMP: JuMP, MOI
using MathOptBenchmarkInstances
using Random
using SparseArrays
using Test

@testset "Checks" begin
    milp, _ = CoolPDLP.random_milp_and_sol(100, 200, 0.4)
    (; c, lv, uv, A, At, lc, uc, D1, D2, int_var) = milp

    @test_nowarn LinearProgram(;
        c, lv, uv, A, At, lc, uc, D1, D2,
    )
    # Type issues
    @test_throws ArgumentError LinearProgram(;
        c = Vector{Any}(c), lv, uv, A, At, lc, uc,
    )
    @test_throws ArgumentError LinearProgram(;
        c = jl(c), lv, uv, A, At, lc, uc,
    )
    # Dimension issues
    @test_throws DimensionMismatch LinearProgram(;
        c = lc, lv, uv, A, At, lc, uc,
    )
    @test_throws DimensionMismatch LinearProgram(;
        c, lv = lc, uv, A, At, lc, uc,
    )
    @test_throws DimensionMismatch LinearProgram(;
        c, lv, uv, A = At, At, lc, uc,
    )
    @test_throws DimensionMismatch LinearProgram(;
        c, lv, uv, A, At, lc = lv, uc,
    )
    @test_throws DimensionMismatch LinearProgram(;
        c, lv, uv, A, At, lc, uc, D1 = D2, D2,
    )
    @test_throws DimensionMismatch LinearProgram(;
        c, lv, uv, A, At, lc, uc, int_var = vcat(int_var, false)
    )
end

@testset "Compare against JuMP" begin
    function jump_nbcons(model)
        eq, ineq = 0, 0
        for (F, S) in JuMP.list_of_constraint_types(model)
            F <: JuMP.AffExpr || continue
            if S <: MOI.EqualTo
                eq += JuMP.num_constraints(model, F, S)
            elseif S <: MOI.GreaterThan || S <: MOI.LessThan || S <: MOI.Interval
                ineq += JuMP.num_constraints(model, F, S)
            else
                error("constraint type not handled")
            end
        end
        return (; eq, ineq)
    end

    netlib = list_instances(Netlib)
    @testset for name in netlib[randperm(length(netlib))[1:20]]
        qps, path = read_instance(Netlib, name)
        milp = LinearProgram(qps; path, name, dataset = "Netlib")
        if name in ["agg", "blend", "dfl001", "forplan", "gfrd-pnc", "sierra"]
            @test_skip JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
        else
            jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
            @test nbvar(milp) == JuMP.num_variables(jump_model)
            @test nbcons_eq(milp) == jump_nbcons(jump_model).eq
            @test nbcons_ineq(milp) == jump_nbcons(jump_model).ineq
        end
    end
end;

@testset "Show" begin
    qps, path = read_instance(Netlib, "seba")
    milp = LinearProgram(qps; path, name = "seba")
    @test startswith(string(milp), "LinearProgram instance seba")
end

@testset "QuadraticProgram Checks" begin
    n, m = 20, 10
    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, 0.4)
    H = sprand(n, n, 0.3)
    Q = H' * H
    lc = rand(m)
    uc = lc + rand(m)

    @test_nowarn QuadraticProgram(; c, lv, uv, A, Q, lc, uc)
    @test_throws DimensionMismatch QuadraticProgram(; c, lv, uv, A, Q = Q[1:end-1, :], lc, uc)
    @test_throws DimensionMismatch QuadraticProgram(; c, lv, uv, A, Q = Q[:, 1:end-1], lc, uc)
    @test_throws ArgumentError QuadraticProgram(; c = Vector{Any}(c), lv, uv, A, Q, lc, uc)
end

@testset "QuadraticProgram Show" begin
    n, m = 5, 3
    c = zeros(n)
    Q = sparse(1.0I, n, n)
    A = spzeros(m, n)
    qp = QuadraticProgram(; c, Q, lv = zeros(n), uv = ones(n), A, lc = zeros(m), uc = ones(m))
    s = string(qp)
    @test startswith(s, "QuadraticProgram")
    @test contains(s, "quadratic: true")  # FIXME: redundant with type name
end

@testset "get_Q" begin
    n, m = 10, 5
    c = rand(n)
    lv = zeros(n)
    uv = ones(n)
    A = sprand(m, n, 0.4)
    lc = zeros(m)
    uc = ones(m)
    lp = LinearProgram(; c, lv, uv, A, lc, uc)
    @test isnothing(CoolPDLP.get_Q(lp))
    Q = sparse(1.0I, n, n)
    qp = QuadraticProgram(; c, lv, uv, A, Q, lc, uc)
    @test CoolPDLP.get_Q(qp) === qp.Q
end
