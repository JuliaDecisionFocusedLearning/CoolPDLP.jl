using CoolPDLP
using CoolPDLP: EachInstance, batched_all, batched_expand, batched_mean, batched_similar,
    batched_zeros, instance, instance_mat, instance_num, instance_vec, nbinstances
using JLArrays
using Random
using Test

Random.seed!(0)

@testset "Instance extraction" begin
    v, m = randn(4), randn(4, 3)
    a = randn(4, 5, 3)

    @test instance_vec(v, 2) === v
    @test instance_vec(m, 2) == m[:, 2]
    @test instance_mat(m, 2) === m
    @test instance_mat(a, 2) == a[:, :, 2]
    @test instance_num(1.5, 2) == 1.5
    @test instance_num([1.0, 2.0, 3.0], 2) == 2.0

    # views, so that instances share the memory of the batch
    instance_vec(m, 2)[1] = 100
    @test m[1, 2] == 100
    instance_mat(a, 2)[1, 1] = 100
    @test a[1, 1, 2] == 100
end

@testset "Per-instance quantities" begin
    v, m = randn(4), randn(4, 3)

    @test batched_expand(v, 2.0) === 2.0
    @test batched_expand(m, 2.0) == fill(2.0, 3)
    @test batched_expand(m, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    @test batched_expand(jl(m), 2.0) isa JLArray{Float64, 1}

    @test batched_similar(2.0) === 2.0
    @test size(batched_similar(zeros(3))) == (3,)

    @test batched_all(>(0), 1.0)
    @test !batched_all(>(0), -1.0)
    @test batched_all(>(0), [1.0, 2.0])
    @test !batched_all(>(0), [1.0, -2.0])

    @test batched_mean(2.0) == 2.0
    @test batched_mean([1.0, 2.0, 3.0]) == 2.0
end

@testset "Batched allocation" begin
    v, m = randn(4), randn(4, 3)

    @test batched_zeros(v, 5, 3, Val(false)) == zeros(5)
    @test batched_zeros(v, 5, 3, Val(true)) == zeros(5, 3)
    @test batched_zeros(m, 5, 3, Val(false)) isa Vector{Float64}
    @test batched_zeros(jl(v), 5, 3, Val(true)) isa JLArray{Float64, 2}
    @test @inferred(batched_zeros(v, 5, 3, Val(false))) isa Vector{Float64}
    @test @inferred(batched_zeros(v, 5, 3, Val(true))) isa Matrix{Float64}
end

@testset "EachInstance of a batched MILP" begin
    nbatch = 3
    milps = [CoolPDLP.random_milp_and_sol(5, 8, 0.5)[1] for _ in 1:nbatch]
    A, int_var = milps[1].A, milps[1].int_var
    milps = map(m -> MILP(; m.c, m.lv, m.uv, A, m.lc, m.uc, int_var), milps)
    stack_batch(f) = stack(f, milps)
    milp_batch = MILP(;
        c = stack_batch(m -> m.c),
        lv = stack_batch(m -> m.lv),
        uv = stack_batch(m -> m.uv),
        A,
        lc = stack_batch(m -> m.lc),
        uc = stack_batch(m -> m.uc),
        int_var,
    )
    each = EachInstance(milp_batch)

    @test eltype(each) == typeof(instance(milp_batch, 1))
    @test size(each) == (nbatch,)
    @test each[2] ≈ milps[2]
    @test all(splat(≈), zip(collect(each), milps))
    @test_throws BoundsError each[nbatch + 1]
end

@testset "Instance counts along the solve" begin
    nbatch = 2
    milp, _ = CoolPDLP.random_milp_and_sol(5, 8, 0.5)
    milp_batch = MILP(;
        c = repeat(milp.c, 1, nbatch),
        lv = repeat(milp.lv, 1, nbatch),
        uv = repeat(milp.uv, 1, nbatch),
        milp.A,
        lc = repeat(milp.lc, 1, nbatch),
        uc = repeat(milp.uc, 1, nbatch),
        milp.int_var,
    )
    sol = PrimalDualSolution(milp_batch)
    @test nbinstances(milp_batch) == nbatch
    @test nbinstances(sol) == nbatch

    @testset "$alg" for alg in (PDHG, PDLP)
        algo = alg(; record_error_history = false)
        state = initialize(milp_batch, sol, algo; starting_time = time())
        @test nbinstances(state) == nbatch
        @test nbinstances(state.scratch) == nbatch
        @test nbinstances(instance(state, 1).sol) == 1
        @test nbinstances(PrimalDualSolution(instance(milp_batch, 1))) == 1
    end
end
