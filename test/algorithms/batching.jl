using CoolPDLP
using CoolPDLP: KKTErrors, Scratch, initialize, instance, kkt_errors!,
    nbinstances, prog_showvalues, relative, restart!, restart_check!, step!,
    termination_status
using MathOptBenchmarkInstances: Netlib, read_instance
using Random
using SparseArrays: sparse
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
    rng = Xoshiro(0)
    milps, milp_batch = random_milp_batch(rng, 20, 30, 0.4, NBATCH; batched)
    # start from a different point in every instance, so that no two columns ever coincide
    xs = [randn(rng, nbvar(milp_batch)) for _ in 1:NBATCH]
    ys = [randn(rng, nbcons(milp_batch)) for _ in 1:NBATCH]
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
        states = map(1:NBATCH) do i
            initialize(milps[i], copy(sols[i]), algo; starting_time = time())
        end
        (; η, ω) = state_batch.step_sizes
        @test length(η) == length(ω) == NBATCH
        for i in 1:NBATCH
            @test η[i] ≈ states[i].step_sizes.η
            @test ω[i] ≈ states[i].step_sizes.ω
            @test instance(state_batch, i).step_sizes.ω ≈ states[i].step_sizes.ω
        end

        # the initial primal weight can be trivial, so also compare after a restart moved it
        ω_init = copy(state_batch.step_sizes.ω)
        for _ in 1:NSTEPS
            step!(state_batch, milp_batch)
            foreach(step!, states, milps)
        end
        restart_check!(state_batch, milp_batch, algo)
        restart!(state_batch, milp_batch, algo)
        for i in 1:NBATCH
            restart_check!(states[i], milps[i], algo)
            restart!(states[i], milps[i], algo)
            @test instance(state_batch, i).step_sizes.ω ≈ states[i].step_sizes.ω
        end
        # guard against a vacuous comparison: the restart must actually move the weights
        @test !(state_batch.step_sizes.ω ≈ ω_init)
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
            @test termination_status(stats) == termination_status(stats_single)
            for i in 1:NBATCH
                @test sol.x[:, i] ≈ sol_single.x
                @test stats.err.primal[i] ≈ stats_single.err.primal
            end
        end
    end
end

@testset "Progress values summarize the batch" begin
    @test CoolPDLP.progress_value(2.5) === 2.5
    # fixed-width values, so `mean` sits at the same column on every row
    @test CoolPDLP.progress_value([1.0, 2.0, 3.0]) == "max 3.000e+00, mean 2.000e+00"

    milps, milp_batch = random_milp_batch(Xoshiro(0), 20, 30, 0.4, NBATCH; batched = (:c,))
    algo = PDLP()
    @testset "batch size $(nbinstances(milp))" for milp in (milps[1], milp_batch)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        kkt_errors!(state.stats.err, state.scratch, state.sol, milp)
        values = prog_showvalues(state)
        @test map(first, values) == ("primal", "dual", "gap")
        for (_, value) in values
            if nbinstances(state) == 1
                @test value isa Real && isfinite(value)
            else
                # a batch shows two summary numbers rather than one value per column
                @test value isa String
                @test occursin("max ", value) && occursin("mean ", value)
            end
        end
    end
end

@testset verbose = true "Terminated instances are frozen" begin
    # instances of very different difficulty: the objective of afiro, scaled per instance
    base = MILP(read_instance(Netlib, "afiro")[1])
    rng = Xoshiro(1)
    scales = (1.0, 1.0e-3, 30.0, 0.5)
    C = stack([s * base.c .* (1 .+ 0.05 .* randn(rng, length(base.c))) for s in scales])
    batch = MILP(; c = C, lv = base.lv, uv = base.uv, A = base.A, lc = base.lc, uc = base.uc)
    B = length(scales)
    @testset "$alg" for alg in (HPDLP, HPRLP)
        algo = alg(; termination_reltol = 1.0e-6, max_kkt_passes = 20_000)
        sol, stats = solve(batch, algo)
        @test termination_status(stats) == CoolPDLP.MOI.OPTIMAL
        passes = [instance(stats, i).kkt_passes for i in 1:B]
        # the batch only ran as long as its slowest instance, the others stopped earlier
        @test maximum(passes) == stats.kkt_passes
        @test minimum(passes) < stats.kkt_passes
        @test !any(stats.active)
        for i in 1:B
            milp_i = MILP(; c = C[:, i], lv = base.lv, uv = base.uv, A = base.A, lc = base.lc, uc = base.uc)
            sol_i, stats_i = solve(milp_i, algo)
            # every instance stops where it would on its own, with the same solution: only the
            # rounding of the batched reductions differs from an unbatched solve
            @test instance(stats, i).kkt_passes == stats_i.kkt_passes
            @test termination_status(instance(stats, i)) == termination_status(stats_i)
            @test sol.x[:, i] ≈ sol_i.x rtol = 1.0e-9
            @test stats.err.primal[i] ≈ stats_i.err.primal rtol = 1.0e-6 atol = 1.0e-12
        end
    end
end

@testset verbose = true "Batch compaction" begin
    base = MILP(read_instance(Netlib, "afiro")[1])
    rng = Xoshiro(2)
    scales = (1.0, 1.0e-3, 30.0, 0.5, 2.0, 1.0e-2, 10.0, 1.0)
    C = stack([s * base.c .* (1 .+ 0.05 .* randn(rng, length(base.c))) for s in scales])
    batch = MILP(; c = C, lv = base.lv, uv = base.uv, A = base.A, lc = base.lc, uc = base.uc)
    B = length(scales)
    @testset "columns and scatter_columns round trip" begin
        algo = HPDLP()
        milp, sol = CoolPDLP.preprocess(batch, PrimalDualSolution(batch), algo)
        state = initialize(milp, sol, algo; starting_time = 0.0)
        idx = [2, 5, 7]
        sub = CoolPDLP.columns(state, idx, B)
        @test typeof(sub) == typeof(state)
        @test sub.sol.x == state.sol.x[:, idx]
        @test sub.stats.active == state.stats.active[idx]
        @test nbinstances(CoolPDLP.columns(milp, idx)) == length(idx)
        sub.sol.x .= -1
        sub.stats.kkt_passes = 123
        CoolPDLP.scatter_columns(state, sub, idx, B)
        @test all(==(-1), state.sol.x[:, idx])
        @test !any(==(-1), state.sol.x[:, setdiff(1:B, idx)])
        @test state.stats.kkt_passes == 123
    end
    # what belongs to the instances is read from the types: sizes cannot tell a batch of `m` or
    # `n` instances from the shared data
    @testset "Batch as large as the number of $dim" for dim in ("constraints", "variables")
        A = sparse([1.0 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 0 1])
        nb = 4
        C = 0.5 .+ rand(Xoshiro(3), 4, nb)
        lp = if dim == "constraints"
            MILP(; c = C, lv = zeros(4), uv = fill(10.0, 4), A, lc = ones(4), uc = fill(5.0, 4))
        else
            # one more constraint, so that only the variables match the batch
            A5 = [A; sparse([1.0 1 1 1])]
            MILP(; c = C, lv = zeros(4), uv = fill(10.0, 4), A = A5, lc = ones(5), uc = fill(5.0, 5))
        end
        idx = [1, 3]
        sub_lp = CoolPDLP.columns(lp, idx)
        @test nbinstances(sub_lp) == length(idx)
        @test sub_lp.c == C[:, idx]
        @test sub_lp.lv == lp.lv && sub_lp.uv == lp.uv
        @test sub_lp.lc == lp.lc && sub_lp.uc == lp.uc
        algo = HPDLP()
        milp, sol = CoolPDLP.preprocess(lp, PrimalDualSolution(lp), algo)
        state = initialize(milp, sol, algo; starting_time = 0.0)
        sub = CoolPDLP.columns(state, idx, nb)
        @test sub.sol.x == state.sol.x[:, idx]
        @test sub.step_sizes.ω == state.step_sizes.ω[idx]
        @test nbinstances(CoolPDLP.columns(milp, idx)) == length(idx)
        (sol0, stats0), (sol1, stats1) = map((false, true)) do batch_compaction
            solve(lp, HPDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 20_000, batch_compaction))
        end
        @test termination_status(stats1) == termination_status(stats0)
        @test sol1.x == sol0.x
    end
    # only PDLP restarts the whole batch at once, from an average over its instances, so it is
    # the only algorithm whose iterates depend on the compaction
    @testset "$alg gives the same result" for alg in (PDHG, HPDLP, HPRLP)
        r = map((false, true)) do batch_compaction
            solve(batch, alg(; termination_reltol = 1.0e-6, max_kkt_passes = 20_000, batch_compaction))
        end
        (sol0, stats0), (sol1, stats1) = r
        @test termination_status(stats1) == termination_status(stats0)
        @test stats1.instance_status_code == stats0.instance_status_code
        @test stats1.instance_kkt_passes == stats0.instance_kkt_passes
        @test stats1.kkt_passes == stats0.kkt_passes
        @test sol1.x == sol0.x
        @test sol1.y == sol0.y
        @test length(stats1.error_history) == length(stats0.error_history)
        @test last(stats1.error_history)[2].primal == last(stats0.error_history)[2].primal
        @test stats1.active == stats0.active
    end
    @testset "PDLP" begin
        sol, stats = solve(batch, PDLP(; termination_reltol = 1.0e-6, max_kkt_passes = 20_000))
        @test termination_status(stats) == CoolPDLP.MOI.OPTIMAL
        @test all(relative(instance(stats, i).err) <= 1.0e-6 for i in 1:B)
    end
end
