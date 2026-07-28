using CoolPDLP
using CoolPDLP: BatchedGPUSparseMatrixCSR, EachInstance, GPUSparseMatrixCSR, KKTErrors, Scratch,
    initialize, instance, kkt_errors!, nbinstances, relative, sametype_transpose, step!
using LinearAlgebra
using Random
using SparseArrays
using Test

const NBATCH = 3
const NSTEPS = 50

"groups of MILP fields which can vary from one instance of a batch to the next"
const BATCHABLE = (:objective, :varbounds, :matrix, :consbounds)

"""
    all_combinations()

Return every subset of `BATCHABLE`, from nothing batched to everything batched.
"""
function all_combinations()
    return map(0:(2^length(BATCHABLE) - 1)) do mask
        Tuple(field for (k, field) in enumerate(BATCHABLE) if isodd(mask >> (k - 1)))
    end
end

combination_name(batched) = isempty(batched) ? "nothing" : join(batched, " + ")

"""
    batched_csr(As)

Stack matrices sharing a single sparsity pattern into one batched CSR matrix.
"""
function batched_csr(As)
    csrs = map(GPUSparseMatrixCSR, As)
    ref = first(csrs)
    @assert all(A -> A.rowptr == ref.rowptr && A.colval == ref.colval, csrs)
    nzval = reduce(hcat, map(A -> A.nzval, csrs))
    return BatchedGPUSparseMatrixCSR(ref.m, ref.n, ref.rowptr, ref.colval, nzval)
end

"""
    make_batch(batched)

Build `NBATCH` single-instance MILPs with their starting points, together with the batched
MILP and starting point which group them.

Only the field groups listed in `batched` vary from one instance to the next: their data is
stored column-wise in the batched MILP, while everything else stays shared.
"""
function make_batch(batched)
    Random.seed!(0)
    m, n = 20, 30
    instances = [CoolPDLP.random_milp_and_sol(m, n, 0.4)[1] for _ in 1:NBATCH]
    # all instances share the sparsity pattern, so the batch needs a single index structure
    pattern = instances[1].A
    As = map(1:NBATCH) do i
        i == 1 && return pattern
        return SparseMatrixCSC(m, n, copy(pattern.colptr), copy(pattern.rowval), randn(nnz(pattern)))
    end
    # field groups outside of `batched` all take their value from the first instance
    vary(group, i) = group in batched ? i : 1
    fields = map(1:NBATCH) do i
        return (
            c = instances[vary(:objective, i)].c,
            lv = instances[vary(:varbounds, i)].lv,
            uv = instances[vary(:varbounds, i)].uv,
            A = As[vary(:matrix, i)],
            lc = instances[vary(:consbounds, i)].lc,
            uc = instances[vary(:consbounds, i)].uc,
        )
    end
    int_var = instances[1].int_var
    milps = map(fields) do f
        MILP(; f.c, f.lv, f.uv, f.A, f.lc, f.uc, int_var)
    end

    stack_batch(getter) = reduce(hcat, map(getter, fields))
    group(g, getter) = g in batched ? stack_batch(getter) : getter(fields[1])
    milp_batch = MILP(;
        c = group(:objective, f -> f.c),
        lv = group(:varbounds, f -> f.lv),
        uv = group(:varbounds, f -> f.uv),
        A = :matrix in batched ? batched_csr(As) : pattern,
        At = :matrix in batched ? batched_csr(map(sametype_transpose, As)) : sametype_transpose(pattern),
        lc = group(:consbounds, f -> f.lc),
        uc = group(:consbounds, f -> f.uc),
        int_var,
    )

    # start from a different point in every instance, so that no two columns ever coincide
    xs = [randn(n) for _ in 1:NBATCH]
    ys = [randn(m) for _ in 1:NBATCH]
    sols = map(PrimalDualSolution, xs, ys)
    # `PrimalDualSolution(milp_batch)` follows the shape of `lv` and `lc`, which are not
    # batched in every combination, so the batched starting point is built explicitly
    sol_batch = PrimalDualSolution(reduce(hcat, xs), reduce(hcat, ys))
    return milps, sols, milp_batch, sol_batch
end

batched_matrix(milp::MILP) = milp.A isa BatchedGPUSparseMatrixCSR

"""
    initialize_batch(milp_batch, sol_batch, algo, milp_single)

Initialize the state of `algo` on `milp_batch`.

`initialize` derives the step size from `spectral_norm(A, At)`, which slices the constraint
matrix instance by instance. On the CPU a slice of a `BatchedGPUSparseMatrixCSR` is a
`SubArray`, which the `GPUSparseMatrixCSR` constructor rejects, so when the matrix is batched
the step size is taken from `milp_single` instead and the single-instance states are aligned
with the result afterwards.
"""
function initialize_batch(milp_batch::MILP, sol_batch, algo, milp_single::MILP)
    milp = if batched_matrix(milp_batch)
        MILP(;
            milp_batch.c, milp_batch.lv, milp_batch.uv,
            A = milp_single.A, At = milp_single.At,
            milp_batch.lc, milp_batch.uc, milp_batch.int_var,
        )
    else
        milp_batch
    end
    return initialize(milp, sol_batch, algo; starting_time = time())
end

@testset verbose = true "Batching $(combination_name(batched))" for batched in all_combinations()
    milps, sols, milp_batch, sol_batch = make_batch(batched)
    nbinst = isempty(batched) ? 1 : NBATCH

    @testset "Batch iteration" begin
        @test nbinstances(milp_batch) == nbinst
        @test length(EachInstance(milp_batch)) == nbinst
        if batched_matrix(milp_batch)
            # slicing a batched CPU matrix is not supported, see `initialize_batch`
            @test_broken instance(milp_batch, 1) isa MILP
        else
            for (i, milp) in enumerate(EachInstance(milp_batch))
                @test milp ≈ milps[i]
            end
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
        if batched_matrix(milp_batch)
            # the step size needs the spectral norm of each instance, see `initialize_batch`
            @test_broken length(
                initialize(milp_batch, copy(sol_batch), algo; starting_time = time()).step_sizes.η
            ) == NBATCH
        else
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
    end

    @testset "Iterates match single solves" begin
        @testset "$alg" for alg in (PDHG, PDLP)
            algo = alg(; record_error_history = false)
            state_batch = initialize_batch(milp_batch, copy(sol_batch), algo, milps[1])
            states = map(1:NBATCH) do i
                state = initialize(milps[i], copy(sols[i]), algo; starting_time = time())
                # the batched step sizes may come from another instance, see `initialize_batch`
                state.step_sizes.η = state_batch.step_sizes.η[i]
                state.step_sizes.ω = state_batch.step_sizes.ω[i]
                return state
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

    @testset "Identical batch matches single solve" begin
        # same shapes as `milp_batch`, but every instance holds the first problem of the batch
        repeat_batch(v) = size(v, 2) == 1 ? v : repeat(view(v, :, 1), 1, NBATCH)
        milp_id = MILP(;
            c = repeat_batch(milp_batch.c),
            lv = repeat_batch(milp_batch.lv),
            uv = repeat_batch(milp_batch.uv),
            A = milps[1].A,
            At = milps[1].At,
            lc = repeat_batch(milp_batch.lc),
            uc = repeat_batch(milp_batch.uc),
            milp_batch.int_var,
        )
        sol_id = PrimalDualSolution(
            repeat(view(sol_batch.x, :, 1), 1, NBATCH),
            repeat(view(sol_batch.y, :, 1), 1, NBATCH),
        )
        @testset "$alg" for alg in (PDHG, PDLP)
            algo = alg(; termination_reltol = 1.0e-6, max_kkt_passes = 2000)
            if batched_matrix(milp_batch)
                # preconditioning expects a two-dimensional constraint matrix
                @test_broken solve(milp_batch, copy(sol_batch), algo) isa Tuple
            else
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
end
