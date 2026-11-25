using CoolPDLP
using Random
using LinearAlgebra
using SparseArrays

"MILP fields which can vary from one instance of a batch to the next"
const BATCHABLE = (:c, :lv, :uv, :lc, :uc)

"""
    random_milp_batch([rng,] m, n, p, nbatch; batched=BATCHABLE)

Build `nbatch` random single MILPs together with the batched MILP grouping them.

Only the fields listed in `batched` vary from one instance to the next: their data is stored
column-wise in the batched MILP, while everything else (including the constraint matrix)
stays shared.
"""
function random_milp_batch(rng::AbstractRNG, m, n, p, nbatch; batched = BATCHABLE)
    instances = [CoolPDLP.random_milp_and_sol(rng, m, n, p)[1] for _ in 1:nbatch]
    A = instances[1].A

    vary(field, i) = field in batched ? i : 1
    fields = map(1:nbatch) do i
        return (
            c = instances[vary(:c, i)].c,
            lv = instances[vary(:lv, i)].lv,
            uv = instances[vary(:uv, i)].uv,
            lc = instances[vary(:lc, i)].lc,
            uc = instances[vary(:uc, i)].uc,
        )
    end
    (; c0, int_var) = instances[1]  # the objective constant is shared by the whole batch
    milps = map(f -> MILP(; f.c, c0, f.lv, f.uv, A, f.lc, f.uc, int_var), fields)

    column(field, getter) = field in batched ? stack(getter, fields) : getter(fields[1])
    milp_batch = MILP(;
        c = column(:c, f -> f.c),
        c0,
        lv = column(:lv, f -> f.lv),
        uv = column(:uv, f -> f.uv),
        A,
        lc = column(:lc, f -> f.lc),
        uc = column(:uc, f -> f.uc),
        int_var,
    )
    return milps, milp_batch
end

function random_milp_batch(m, n, p, nbatch; kwargs...)
    return random_milp_batch(Random.default_rng(), m, n, p, nbatch; kwargs...)
end

"""
    same_instance(m1, m2)

Compare two MILPs which may hold their constraint matrix in different formats.
"""
function same_instance(m1::MILP, m2::MILP)
    as_csc(milp) = MILP(;
        milp.c, milp.c0, milp.lv, milp.uv,
        A = SparseMatrixCSC(milp.A), At = SparseMatrixCSC(milp.At),
        milp.lc, milp.uc, milp.D1, milp.D2, milp.int_var, milp.var_names, milp.con_names,
        milp.dataset, milp.name, milp.path,
    )
    return as_csc(m1) ≈ as_csc(m2)
end

"""
    random_qp(rng, m, n; density, rank)

Build a random convex QP which is feasible by construction (its bounds surround a random point)
and bounded (`Q` is positive definite unless `rank < n`, in which case every variable is boxed).
"""
function random_qp(rng, m, n; density = 0.3, rank = n)
    B = sprandn(rng, rank, n, density)
    Q = sparse(B' * B)
    rank == n && (Q += 0.1I)
    A = sprandn(rng, m, n, density)
    x0 = randn(rng, n)
    boxed = rank < n
    lv = [boxed || rand(rng) < 0.7 ? x - rand(rng) : -Inf for x in x0]
    uv = [boxed || rand(rng) < 0.7 ? x + rand(rng) : Inf for x in x0]
    Ax0 = A * x0
    lc = [rand(rng) < 0.3 ? a : (rand(rng) < 0.5 ? a - rand(rng) : -Inf) for a in Ax0]
    uc = [lc[i] == Ax0[i] ? Ax0[i] : (rand(rng) < 0.7 ? Ax0[i] + rand(rng) : Inf) for i in eachindex(Ax0)]
    c = randn(rng, n)
    return MILP(; c, c0 = randn(rng), Q, lv, uv, A, lc, uc)
end
