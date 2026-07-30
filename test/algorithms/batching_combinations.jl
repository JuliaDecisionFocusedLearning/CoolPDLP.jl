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
    # unlike the zero solution `PrimalDualSolution(milp_batch)`, this starting point holds a
    # different value in every column
    sol_batch = PrimalDualSolution(reduce(hcat, xs), reduce(hcat, ys))
    return milps, sols, milp_batch, sol_batch
end

batched_matrix(milp::MILP) = milp.A isa BatchedGPUSparseMatrixCSR

"""
    same_instance(m1, m2)

Compare two MILPs which may hold their constraint matrix in different formats.
"""
function same_instance(m1::MILP, m2::MILP)
    return (
        m1.c ≈ m2.c &&
            m1.lv ≈ m2.lv &&
            m1.uv ≈ m2.uv &&
            SparseMatrixCSC(m1.A) ≈ SparseMatrixCSC(m2.A) &&
            SparseMatrixCSC(m1.At) ≈ SparseMatrixCSC(m2.At) &&
            m1.lc ≈ m2.lc &&
            m1.uc ≈ m2.uc &&
            m1.int_var == m2.int_var
    )
end

@testset verbose = true "Batching $(combination_name(batched))" for batched in all_combinations()
    milps, sols, milp_batch, sol_batch = make_batch(batched)
    nbinst = isempty(batched) ? 1 : NBATCH

    @testset "Batch iteration" begin
        @test nbinstances(milp_batch) == nbinst
        @test length(EachInstance(milp_batch)) == nbinst
        @test nbinstances(PrimalDualSolution(milp_batch)) == nbinst
        for (i, milp) in enumerate(EachInstance(milp_batch))
            @test same_instance(milp, milps[i])
        end
    end

    @testset "Display" begin
        str = sprint(show, milp_batch)
        @test startswith(str, "MILP instance")
        # the counts describe one instance, not the whole batch
        @test occursin("- constraints: $(nbcons(milps[1]))", str)
        @test occursin("- nonzeros: $(nnz(milps[1].A))", str)
        if :consbounds in batched
            @test !occursin("equalities", str)
            @test_throws ArgumentError nbcons_eq(milp_batch)
        else
            @test occursin("- $(nbcons_eq(milps[1])) equalities", str)
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
                # a batch shares a single preconditioner, so it needs a single matrix
                @test_throws "batched constraint matrices" solve(milp_batch, copy(sol_batch), algo)
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
