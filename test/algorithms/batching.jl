using CoolPDLP
using CoolPDLP: KKTErrors, Scratch, initialize,
    instance, kkt_errors!, nbinstances, prog_showvalues, relative, step!
using Random
using Test

const NBATCH = 3
const NSTEPS = 50

"Return every subset of `BATCHABLE`, from nothing batched to everything batched."
function all_combinations()
    # bit k of the mask selects `BATCHABLE[k]`
    return map(0:(2^length(BATCHABLE) - 1)) do mask
        Tuple(field for (k, field) in enumerate(BATCHABLE) if isodd(mask >> (k - 1)))
    end
end

combination_name(batched) = isempty(batched) ? "nothing" : join(batched, " + ")

"""
    paired_bounds(batched)

Return whether each pair of bounds is batched as a whole.

The full solve pipeline is only compared against single solves on such a combination: it
starts from a batch of identical instances, where splitting a pair changes nothing but costs
a specialization of the whole solver.
"""
function paired_bounds(batched)
    return (:lv in batched) == (:uv in batched) && (:lc in batched) == (:uc in batched)
end

function make_batch(batched)
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, NBATCH; batched)
    # start from a different point in every instance, so that no two columns ever coincide
    xs = [randn(nbvar(milp_batch)) for _ in 1:NBATCH]
    ys = [randn(nbcons(milp_batch)) for _ in 1:NBATCH]
    sols = map(PrimalDualSolution, xs, ys)
    return milps, sols, milp_batch, PrimalDualSolution(stack(xs), stack(ys))
end

@testset verbose = true "Batching $(combination_name(batched))" for batched in all_combinations()
    milps, sols, milp_batch, sol_batch = make_batch(batched)
    nbinst = isempty(batched) ? 1 : NBATCH

    @testset "Batch iteration" begin
        @test nbinstances(milp_batch) == nbinst
        @test nbinstances(PrimalDualSolution(milp_batch)) == nbinst
        for i in 1:nbinst
            @test same_instance(instance(milp_batch, i), milps[i])
        end
    end

    @testset "KKT errors per instance" begin
        err_batch = kkt_errors!(KKTErrors(sol_batch), Scratch(sol_batch), sol_batch, milp_batch)
        @test err_batch.primal isa Vector{Float64}
        @test length(err_batch.primal) == NBATCH
        # guard against a vacuous comparison: the instances must not all be the same problem
        @test allunique(err_batch.primal)
        for i in 1:NBATCH
            err = kkt_errors!(KKTErrors(sols[i]), Scratch(sols[i]), sols[i], milps[i])
            @test instance(err_batch, i) ≈ err
            @test relative(err_batch)[i] ≈ relative(err)
        end
    end

    @testset "Step sizes per instance" begin
        algo = PDLP()
        state_batch = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
        (; η, ω) = state_batch.step_sizes
        @test length(η) == length(ω) == NBATCH
        for i in 1:NBATCH
            state = initialize(milps[i], copy(sols[i]), algo; starting_time = time())
            @test η[i] ≈ state.step_sizes.η
            @test ω[i] ≈ state.step_sizes.ω
            @test instance(state_batch, i).step_sizes.ω ≈ state.step_sizes.ω
        end
    end

    @testset "Iterates match single solves" begin
        @testset "$alg" for alg in (PDHG, PDLP)
            algo = alg(; record_error_history = false)
            state_batch = initialize(milp_batch, copy(sol_batch), algo; starting_time = time())
            states = map(1:NBATCH) do i
                initialize(milps[i], copy(sols[i]), algo; starting_time = time())
            end
            for _ in 1:NSTEPS
                step!(state_batch, milp_batch)
                for i in 1:NBATCH
                    step!(states[i], milps[i])
                end
            end
            # guard against a vacuous comparison: no two instances should coincide
            @test allunique(eachcol(state_batch.sol.x))
            for i in 1:NBATCH
                @test instance(state_batch, i).sol ≈ states[i].sol
            end
        end
    end

    paired_bounds(batched) && @testset "Identical batch matches single solve" begin
        # same shapes as `milp_batch`, but every instance holds the first problem of the batch
        repeat_batch(v) = size(v, 2) == 1 ? v : repeat(view(v, :, 1), 1, NBATCH)
        milp_id = MILP(;
            c = repeat_batch(milp_batch.c),
            lv = repeat_batch(milp_batch.lv),
            uv = repeat_batch(milp_batch.uv),
            A = milps[1].A,
            lc = repeat_batch(milp_batch.lc),
            uc = repeat_batch(milp_batch.uc),
            milp_batch.int_var,
        )
        sol_id = PrimalDualSolution(
            repeat(view(sol_batch.x, :, 1), 1, NBATCH),
            repeat(view(sol_batch.y, :, 1), 1, NBATCH),
        )
        @testset "$alg" for alg in (PDHG, PDLP)
            algo = alg(; max_kkt_passes = 200)
            sol, stats = solve(milp_id, sol_id, algo)
            sol_single, stats_single = solve(milps[1], sols[1], algo)
            @test stats.kkt_passes == stats_single.kkt_passes
            @test stats.termination_status == stats_single.termination_status
            for i in 1:NBATCH
                @test sol.x[:, i] ≈ sol_single.x
                @test stats.err.primal[i] ≈ stats_single.err.primal
            end
        end
    end
end

@testset "Progress values per column" begin
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, NBATCH; batched = (:c,))
    algo = PDLP()
    @testset "batch size $(nbinstances(milp))" for milp in (milps[1], milp_batch)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        kkt_errors!(state.stats.err, state.scratch, state.sol, milp)
        values = prog_showvalues(state)
        @test map(first, values) == ("primal", "dual", "gap")
        for (_, value) in values
            @test length(value) == nbinstances(state)
            @test all(isfinite.(value))
        end
    end
end
