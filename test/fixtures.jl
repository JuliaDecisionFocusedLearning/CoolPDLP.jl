using CoolPDLP
using CoolPDLP: BatchedGPUSparseMatrixCSR
using Random
using SparseArrays

"MILP fields which can vary from one instance of a batch to the next"
const BATCHABLE = (:c, :lv, :uv, :A, :lc, :uc)

"""
    random_milp_batch(m, n, p, nbatch; batched=BATCHABLE)

Build `nbatch` random single MILPs together with the batched MILP grouping them.

Only the fields listed in `batched` vary from one instance to the next: their data is stored
column-wise in the batched MILP, while everything else stays shared.
"""
function random_milp_batch(m, n, p, nbatch; batched = BATCHABLE)
    instances = [CoolPDLP.random_milp_and_sol(m, n, p)[1] for _ in 1:nbatch]
    pattern = instances[1].A
    As = map(i -> i == 1 ? pattern : same_pattern(pattern), 1:nbatch)

    vary(field, i) = field in batched ? i : 1
    fields = map(1:nbatch) do i
        return (
            c = instances[vary(:c, i)].c,
            lv = instances[vary(:lv, i)].lv,
            uv = instances[vary(:uv, i)].uv,
            A = As[vary(:A, i)],
            lc = instances[vary(:lc, i)].lc,
            uc = instances[vary(:uc, i)].uc,
        )
    end
    int_var = instances[1].int_var
    milps = map(f -> MILP(; f.c, f.lv, f.uv, f.A, f.lc, f.uc, int_var), fields)

    column(field, getter) = field in batched ? stack(getter, fields) : getter(fields[1])
    milp_batch = MILP(;
        c = column(:c, f -> f.c),
        lv = column(:lv, f -> f.lv),
        uv = column(:uv, f -> f.uv),
        A = :A in batched ? BatchedGPUSparseMatrixCSR(As) : pattern,
        lc = column(:lc, f -> f.lc),
        uc = column(:uc, f -> f.uc),
        int_var,
    )
    return milps, milp_batch
end

"""
    same_pattern(A, [rng])

Return a sparse matrix with the same sparsity pattern as `A` but fresh values, as required by a batched matrix.
"""
function same_pattern(A::SparseMatrixCSC, rng = Random.default_rng())
    return SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), randn(rng, nnz(A)))
end

"""
    same_instance(m1, m2)

Compare two MILPs which may hold their constraint matrix in different formats.
"""
function same_instance(m1::MILP, m2::MILP)
    as_csc(milp) = MILP(;
        milp.c, milp.lv, milp.uv,
        A = SparseMatrixCSC(milp.A), At = SparseMatrixCSC(milp.At),
        milp.lc, milp.uc, milp.D1, milp.D2, milp.int_var, milp.var_names,
        milp.dataset, milp.name, milp.path,
    )
    return as_csc(m1) ≈ as_csc(m2)
end
