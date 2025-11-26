using CoolPDLP
using JLArrays
using JuMP: JuMP, MOI
using MathProgBenchmarks
using Random
using SparseArrays
using Test

@testset "Checks" begin
    m, n = 10, 20
    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, 0.3)
    At = sparse(transpose(A))
    lc = rand(m)
    uc = lc + rand(m)
    D1 = Diagonal(ones(m))
    D2 = Diagonal(ones(n))
    int_var = rand(Bool, n)
    var_names = map(string, 1:n)
    dataset = "dataset"
    name = "name"
    path = "path"

    @test_nowarn MILP(;
        c, lv, uv, A, At, lc, uc, D1, D2,
    )
    # Type issues
    @test_throws ArgumentError MILP(;
        c = Vector{Any}(c), lv, uv, A, At, lc, uc,
    )
    @test_throws ArgumentError MILP(;
        c = jl(c), lv, uv, A, At, lc, uc,
    )
    # Dimension issues
    @test_throws DimensionMismatch MILP(;
        c = lc, lv, uv, A, At, lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv = lc, uv, A, At, lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A = At, At, lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A, At, lc = lv, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A, At, lc, uc, D1 = D2, D2,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A, At, lc, uc, int_var = vcat(int_var, false)
    )
end

@testset "Counting" begin
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
        milp = MILP(qps; path, name, dataset = "Netlib")
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
    milp = MILP(qps; path, name = "seba")
    @test startswith(string(milp), "MILP instance seba")
end
