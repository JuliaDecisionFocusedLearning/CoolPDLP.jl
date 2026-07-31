using CoolPDLP
using CoolPDLP: EachInstance, KKTErrors, Scratch, initialize, instance, kkt_errors!,
    nbinstances, preprocess, relative, step!
using GPUArraysCore: @allowscalar
using Random
using Test

include("../fixtures.jl")

"""
    test_batching(matrix_type, backend; nbatch = 3)

Check that a batch of problems living on `backend` behaves like the same problems handled one at a time.
"""
function test_batching(matrix_type, backend; nbatch = 3)
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(
        20, 30, 0.4, nbatch; batched = filter(!=(:A), BATCHABLE)
    )

    algo = PDHG(Float64, Int, matrix_type; backend)
    # the preconditioner only depends on `A`, so the batch and the single problems share it
    to_device(milp) = preprocess(milp, PrimalDualSolution(milp), algo)
    milp_dev, sol_dev = to_device(milp_batch)
    singles = map(to_device, milps)

    @testset "Instance iteration" begin
        @test nbinstances(milp_dev) == nbatch
        @test length(EachInstance(milp_dev)) == nbatch
        for (i, milp_i) in enumerate(EachInstance(milp_dev))
            milp_single = singles[i][1]
            @test Array(milp_i.c) ≈ Array(milp_single.c)
            @test Array(milp_i.lv) ≈ Array(milp_single.lv)
            @test Array(milp_i.uv) ≈ Array(milp_single.uv)
            @test Array(milp_i.lc) ≈ Array(milp_single.lc)
            @test Array(milp_i.uc) ≈ Array(milp_single.uc)
        end
    end

    @testset "KKT errors per instance" begin
        err_batch = kkt_errors!(KKTErrors(sol_dev), Scratch(sol_dev), sol_dev, milp_dev)
        @test length(err_batch.primal) == nbatch
        rel_batch = Array(relative(err_batch))
        for i in 1:nbatch
            milp_single, sol_single = singles[i]
            err = kkt_errors!(
                KKTErrors(sol_single), Scratch(sol_single), sol_single, milp_single
            )
            @test Array(err_batch.primal)[i] ≈ err.primal
            @test Array(err_batch.dual)[i] ≈ err.dual
            @test Array(err_batch.gap)[i] ≈ err.gap
            @test rel_batch[i] ≈ relative(err)
            @allowscalar @test instance(err_batch, i) ≈ err
        end
    end

    @testset "Iterates match single solves" begin
        state_batch = initialize(milp_dev, copy(sol_dev), algo; starting_time = time())
        states = map(singles) do (milp_single, sol_single)
            initialize(milp_single, copy(sol_single), algo; starting_time = time())
        end
        for _ in 1:50
            step!(state_batch, milp_dev)
            for (i, state) in enumerate(states)
                step!(state, singles[i][1])
            end
        end
        for i in 1:nbatch
            @test Array(state_batch.sol.x[:, i]) ≈ Array(states[i].sol.x) rtol = 1.0e-6
            @test Array(state_batch.sol.y[:, i]) ≈ Array(states[i].sol.y) rtol = 1.0e-6
            @allowscalar @test instance(state_batch, i).step_sizes.ω ≈
                states[i].step_sizes.ω
        end
    end

    @testset "Identical batch matches single solve" begin
        repeat_batch(v) = repeat(v, 1, nbatch)
        milp_id = MILP(;
            c = repeat_batch(milps[1].c),
            lv = repeat_batch(milps[1].lv),
            uv = repeat_batch(milps[1].uv),
            milps[1].A,
            lc = repeat_batch(milps[1].lc),
            uc = repeat_batch(milps[1].uc),
            milps[1].int_var,
        )
        @testset "$alg" for alg in (PDHG, PDLP)
            algo_solve = alg(Float64, Int, matrix_type; backend, max_kkt_passes = 200)
            sol, stats = solve(milp_id, algo_solve)
            sol_single, stats_single = solve(milps[1], algo_solve)
            @test stats.termination_status == stats_single.termination_status
            x, x_single = Array(sol.x), Array(sol_single.x)
            obj_single = objective_value(x_single, milps[1])
            for i in 1:nbatch
                @test objective_value(x[:, i], milps[1]) ≈ obj_single rtol = 1.0e-3
                @test Array(stats.err.primal)[i] ≈ stats_single.err.primal rtol = 1.0e-3
            end
        end
    end

    return nothing
end
