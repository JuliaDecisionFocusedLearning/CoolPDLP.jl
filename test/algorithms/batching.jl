using CoolPDLP
using CoolPDLP: EachInstance, KKTErrors, Scratch, instance, nbinstances, initialize, kkt_errors!,
    primal_weight_update!!, prog_showvalues, relative, step!
using LinearAlgebra
using Random
using SparseArrays
using Test

Random.seed!(0)

# several problems sharing the same constraint matrix, but with their own objective and bounds
nbatch = 3
milps_init = [CoolPDLP.random_milp_and_sol(20, 30, 0.4)[1] for _ in 1:nbatch]
A, int_var = milps_init[1].A, milps_init[1].int_var
milps = map(milps_init) do m
    MILP(; c = m.c, lv = m.lv, uv = m.uv, A, lc = m.lc, uc = m.uc, int_var)
end
sols = map(PrimalDualSolution, milps)

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
sol_batch = PrimalDualSolution(milp_batch)

@testset "Batch iteration" begin
    @test nbinstances(milp_batch) == nbatch
    @test length(EachInstance(milp_batch)) == nbatch
    for (i, milp) in enumerate(EachInstance(milp_batch))
        @test milp ≈ milps[i]
    end
end

@testset "KKT errors per column" begin
    err_batch = kkt_errors!(KKTErrors(sol_batch), Scratch(sol_batch), sol_batch, milp_batch)
    @test err_batch.primal isa Vector{Float64}
    @test length(err_batch.primal) == nbatch
    for i in 1:nbatch
        err = kkt_errors!(KKTErrors(sols[i]), Scratch(sols[i]), sols[i], milps[i])
        @test instance(err_batch, i) ≈ err
        @test relative(err_batch)[i] ≈ relative(err)
    end
end

@testset "Step sizes per column" begin
    algo = PDLP()
    state_batch = initialize(milp_batch, sol_batch, algo; starting_time = time())
    (; η, ω) = state_batch.step_sizes
    @test length(η) == length(ω) == nbatch
    for i in 1:nbatch
        state = initialize(milps[i], sols[i], algo; starting_time = time())
        @test η[i] ≈ state.step_sizes.η
        @test ω[i] ≈ state.step_sizes.ω
        @test instance(state_batch, i).step_sizes.ω ≈ state.step_sizes.ω
    end
end

@testset "Primal weight left alone without movement" begin
    algo = PDLP()
    state = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
    ω = copy(state.step_sizes.ω)
    # the candidate sits exactly on the restart point, so there is nothing to learn from it
    updated = primal_weight_update!!(
        state.scratch, state.step_sizes, state.sol, state.sol, algo.step_size
    )
    @test updated == ω
end

@testset "Progress values per column" begin
    algo = PDLP()
    @testset "batch size $(size(sol.x, 2))" for (milp, sol) in
        ((milps[1], sols[1]), (milp_batch, sol_batch))
        state = initialize(milp, copy(sol), algo; starting_time = time())
        kkt_errors!(state.stats.err, state.scratch, state.sol, milp)
        values = prog_showvalues(state)
        @test map(first, values) == ("primal", "dual", "gap")
        for (_, value) in values
            @test length(value) == nbinstances(state)
            @test all(isfinite.(value))
        end
    end
end

@testset "Bounds batched on their own" begin
    # only the variable lower bound and the constraint upper bound vary across the batch
    shared = milps[1]
    milps_asym = map(milps_init) do m
        MILP(; shared.c, m.lv, shared.uv, A, shared.lc, m.uc, int_var)
    end
    milp_asym = MILP(;
        shared.c,
        lv = stack(m -> m.lv, milps_init),
        shared.uv,
        A,
        shared.lc,
        uc = stack(m -> m.uc, milps_init),
        int_var,
    )

    @test nbinstances(milp_asym) == nbatch
    for (i, milp) in enumerate(EachInstance(milp_asym))
        @test milp ≈ milps_asym[i]
    end

    algo = PDHG()
    state_batch = initialize(
        milp_asym, PrimalDualSolution(milp_asym), algo; starting_time = time()
    )
    states = map(milps_asym) do milp
        initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
    end
    for _ in 1:100
        step!(state_batch, milp_asym)
        for i in 1:nbatch
            step!(states[i], milps_asym[i])
        end
    end
    for i in 1:nbatch
        @test instance(state_batch, i).sol ≈ states[i].sol
    end
end

@testset "Iterates match single solves" begin
    algo = PDHG()
    state_batch = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
    states = map(1:nbatch) do i
        initialize(milps[i], copy(sols[i]), algo; starting_time = time())
    end
    for _ in 1:100
        step!(state_batch, milp_batch)
        for i in 1:nbatch
            step!(states[i], milps[i])
        end
    end
    for i in 1:nbatch
        @test instance(state_batch, i).sol ≈ states[i].sol
    end
end

@testset "Identical batch matches single solve" begin
    milp_id = MILP(;
        c = repeat(milps[1].c, 1, nbatch),
        lv = repeat(milps[1].lv, 1, nbatch),
        uv = repeat(milps[1].uv, 1, nbatch),
        A,
        lc = repeat(milps[1].lc, 1, nbatch),
        uc = repeat(milps[1].uc, 1, nbatch),
        int_var,
    )
    @testset "$alg" for alg in (PDHG, PDLP)
        algo = alg(; termination_reltol = 1.0e-6, max_kkt_passes = 2000)
        sol, stats = solve(milp_id, algo)
        sol_single, stats_single = solve(milps[1], algo)
        @test stats.kkt_passes == stats_single.kkt_passes
        @test stats.termination_status == stats_single.termination_status
        for i in 1:nbatch
            @test sol.x[:, i] ≈ sol_single.x
            @test stats.err.primal[i] ≈ stats_single.err.primal
        end
    end
end
