using CoolPDLP
using CoolPDLP: BatchedGPUSparseMatrixCSR, GPUSparseMatrixCSR, instance
using JLArrays
using JuMP: JuMP, MOI
using MathOptBenchmarkInstances
using Random
using SparseArrays
using Test

@testset "Checks" begin
    milp, _ = CoolPDLP.random_milp_and_sol(100, 200, 0.4)
    (; c, lv, uv, A, At, lc, uc, D1, D2, int_var) = milp

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
    # Batch size issues
    @test_nowarn MILP(;
        c = repeat(c, 1, 3), lv, uv, A, At, lc = repeat(lc, 1, 3), uc = repeat(uc, 1, 3),
    )
    @test_throws DimensionMismatch MILP(;
        c = repeat(c, 1, 3), lv, uv, A, At, lc = repeat(lc, 1, 2), uc = repeat(uc, 1, 2),
    )
    @test_throws DimensionMismatch MILP(;
        c, lv = repeat(lv, 1, 3), uv = repeat(uv, 1, 2), A, At, lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A, At, lc = repeat(lc, 1, 3), uc = repeat(uc, 1, 2),
    )
    # the batch dimension of the constraint matrix counts too
    stack_csr(M, nb) = (
        csr = GPUSparseMatrixCSR(M);
        BatchedGPUSparseMatrixCSR(
            csr.m, csr.n, csr.rowptr, csr.colval, repeat(csr.nzval, 1, nb)
        )
    )
    @test_nowarn MILP(;
        c = repeat(c, 1, 3), lv, uv, A = stack_csr(A, 3), At = stack_csr(At, 3), lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c = repeat(c, 1, 5), lv, uv, A = stack_csr(A, 3), At = stack_csr(At, 3), lc, uc,
    )
    @test_throws DimensionMismatch MILP(;
        c, lv, uv, A = stack_csr(A, 3), At = stack_csr(At, 2), lc, uc,
    )
end

@testset "Batched objective value" begin
    nbatch = 3
    milp, _ = CoolPDLP.random_milp_and_sol(10, 20, 0.4)
    milp_batch = MILP(;
        c = reduce(hcat, [k .* milp.c for k in 1:nbatch]),
        milp.lv, milp.uv, milp.A, milp.lc, milp.uc, milp.int_var,
    )
    x = randn(nbvar(milp), nbatch)

    obj = objective_value(x, milp_batch)
    @test obj isa Vector{Float64}
    @test length(obj) == nbatch
    for i in 1:nbatch
        @test obj[i] ≈ objective_value(x[:, i], instance(milp_batch, i))
    end
end

@testset "Batched constraint counts" begin
    nbatch = 3
    A = sparse([1.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 -1.0])
    lc, uc = [1.0, 2.0, -Inf, 0.0], [1.0, 2.0, 5.0, 3.0]
    c, lv, uv = [1.0, 1.0], zeros(2), fill(10.0, 2)
    milp = MILP(; c, lv, uv, A, lc, uc)
    milp_obj = MILP(; c = repeat(c, 1, nbatch), lv, uv, A, lc, uc)
    milp_cons = MILP(;
        c, lv, uv, A, lc = repeat(lc, 1, nbatch), uc = repeat(uc, 1, nbatch),
    )

    @test nbcons(milp) == nbcons(milp_obj) == nbcons(milp_cons) == 4
    @test nbcons_eq(milp) == nbcons_eq(milp_obj) == 2
    @test nbcons_ineq(milp) == nbcons_ineq(milp_obj) == 2
    # the split between equalities and inequalities may vary across a batch
    @test_throws ArgumentError nbcons_eq(milp_cons)
    @test_throws ArgumentError nbcons_ineq(milp_cons)
    @test !occursin("equalities", string(milp_cons))
    @test occursin("2 equalities", string(milp_obj))
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

@testset "Approx" begin
    netlib = list_instances(Netlib)
    qps, path = read_instance(Netlib, netlib[1])
    milp = MILP(qps; path, dataset = "Netlib")
    @test milp ≈ milp
end
