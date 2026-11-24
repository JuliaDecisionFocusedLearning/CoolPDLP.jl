#=
Batch compaction

A frozen instance of a batch (see `freeze_terminated!`) still pays for its columns of the sparse
products, so once most of a batch has terminated, the slowest instances drag the finished ones
along until the end. On the host (CPU arrays, outside any Reactant compilation), the solve loops
call `maybe_compact!` after every check: once at least half of a batch of at least
`COMPACTION_MIN_BATCH` instances is frozen, the instances still running are copied into a smaller
batch of the same type, solved to the end (compacting again if need be), and scattered back.

The copies are made field by field (`columns`, `scatter_columns`). What belongs to the instances is
decided from the types, never from the sizes, which cannot tell a vector of one entry per instance
from one of `m` entries when the batch has `m` instances: in a batched state, every dense `Matrix`
holds one column per instance and every dense `Vector` of numbers one entry per instance, while
everything else (sparse matrices, counters, parameters) is shared. A dense array which is shared
nonetheless is declared by the type which holds it, in `compaction_shared_fields`, and an array
whose size does not match the batch is an error rather than silently kept or split. The fields of
a `MILP` are restricted by `columns(milp, idx)`, which knows which of them are batched.

A compiled Reactant solve never compacts, since that would change the shapes of the compiled
program: `maybe_compact!` only acts on a host `Bool` stop flag and `Array` storage.
=#

"Smallest batch which [`maybe_compact!`](@ref) compacts."
const COMPACTION_MIN_BATCH = 4

"""
    compaction_shared_fields(x)

Names of the fields of `x` which hold vectors shared by all the instances of a batch, and must not
be split by [`columns`](@ref) (see the comment at the top of `compaction.jl`).
"""
compaction_shared_fields(::Any) = ()

"""
    columns(x, idx, nb)

Return a copy of `x`, a part of the state of a batch of `nb` instances, restricted to the instances
`idx`. Shared parts are kept as they are; see the comment at the top of `compaction.jl`.
"""
columns(x::Union{Number, Symbol, Nothing, Val, Function, AbstractString}, ::Vector{Int}, ::Int) = x
columns(x::AbstractArray, ::Vector{Int}, ::Int) = x
columns(x::Matrix{<:Number}, idx::Vector{Int}, nb::Int) = check_batch_size(x, nb)[:, idx]
columns(x::Vector{<:Number}, idx::Vector{Int}, nb::Int) = check_batch_size(x, nb)[idx]

"Throw unless the last dimension of `x`, a part of a batched state, has one entry per instance."
function check_batch_size(x::AbstractVecOrMat, nb::Int)
    size(x, ndims(x)) == nb || throw(
        DimensionMismatch(
            "a $(typeof(x)) of size $(size(x)) in a batch of $nb instances: declare it in `compaction_shared_fields` if it is shared",
        ),
    )
    return x
end
# the error history of the sub-batch starts empty: see the matching `scatter_columns`
columns(x::Vector{<:Tuple{Number, KKTErrors}}, ::Vector{Int}, ::Int) = similar(x, 0)
columns(x::Union{Tuple, NamedTuple}, idx::Vector{Int}, nb::Int) = map(v -> columns(v, idx, nb), x)
function columns(x, idx::Vector{Int}, nb::Int)
    T = typeof(x)
    shared = compaction_shared_fields(x)
    vals = ntuple(Val(fieldcount(T))) do i
        v = getfield(x, i)
        fieldname(T, i) in shared ? v : columns(v, idx, nb)
    end
    return T(vals...)
end

"""
    scatter_columns(dest, src, idx, nb)

Write `src`, built by [`columns`](@ref) from `dest` and since updated, back into the instances
`idx` of `dest`, and return the merged value: `dest` itself for arrays and mutable structs, which
are updated in place, and `src` for the shared numbers (counters, clocks).
"""
scatter_columns(::Any, src::Union{Number, Symbol, Nothing, Val, Function, AbstractString}, ::Vector{Int}, ::Int) = src
scatter_columns(::AbstractArray, src::AbstractArray, ::Vector{Int}, ::Int) = src
function scatter_columns(dest::Matrix{<:Number}, src::Matrix{<:Number}, idx::Vector{Int}, nb::Int)
    check_batch_size(dest, nb)[:, idx] .= src
    return dest
end
function scatter_columns(dest::Vector{<:Number}, src::Vector{<:Number}, idx::Vector{Int}, nb::Int)
    check_batch_size(dest, nb)[idx] .= src
    return dest
end
# each entry of the sub-batch is completed with the errors of the frozen instances, which have not
# changed since the last entry of `dest`
function scatter_columns(
        dest::Vector{<:Tuple{Number, KKTErrors}}, src::Vector{<:Tuple{Number, KKTErrors}},
        idx::Vector{Int}, nb::Int,
    )
    isempty(src) && return dest
    base = last(dest)[2]
    for (passes, err) in src
        push!(dest, (passes, scatter_columns(copy(base), err, idx, nb)))
    end
    return dest
end
function scatter_columns(dest::Union{Tuple, NamedTuple}, src::Union{Tuple, NamedTuple}, idx::Vector{Int}, nb::Int)
    return map((d, s) -> scatter_columns(d, s, idx, nb), dest, src)
end
function scatter_columns(dest, src, idx::Vector{Int}, nb::Int)
    T = typeof(dest)
    shared = compaction_shared_fields(dest)
    vals = ntuple(Val(fieldcount(T))) do i
        d, s = getfield(dest, i), getfield(src, i)
        fieldname(T, i) in shared ? s : scatter_columns(d, s, idx, nb)
    end
    if ismutabletype(T)
        ntuple(Val(fieldcount(T))) do i
            # a `const` field holds an array, which has been updated in place
            isconst(T, i) || setfield!(dest, i, vals[i])
            nothing
        end
        return dest
    else
        return T(vals...)
    end
end

"""
    columns(milp::MILP, idx)

Restrict a batched `milp` to the instances `idx`, keeping the shared data (`A`, `Q`, the
preconditioner, the bounds common to all instances) as is. A vector of `c`, `lv`, `uv`, `lc` or
`uc` is shared by all the instances and a matrix has one column per instance, whatever their
sizes; `bound_scale` and `obj_scale` are numbers when shared and vectors otherwise.
"""
function columns(milp::MILP, idx::Vector{Int})
    nb = nbinstances(milp)
    return MILP(;
        c = milp_columns(milp.c, idx, nb),
        c0 = milp.c0,
        lv = milp_columns(milp.lv, idx, nb),
        uv = milp_columns(milp.uv, idx, nb),
        A = milp.A,
        At = milp.At,
        Q = columns_quadratic(milp.Q, idx, nb),
        lc = milp_columns(milp.lc, idx, nb),
        uc = milp_columns(milp.uc, idx, nb),
        D1 = milp.D1,
        D2 = milp.D2,
        bound_scale = milp_scale_columns(milp.bound_scale, idx, nb),
        obj_scale = milp_scale_columns(milp.obj_scale, idx, nb),
        int_var = milp.int_var,
        var_names = milp.var_names,
        con_names = milp.con_names,
        dataset = milp.dataset,
        name = milp.name,
        path = milp.path
    )
end

"""
    milp_columns(x, idx, nb)

Restrict a data field of a batched `MILP` to the instances `idx`: the columns of a matrix (one
per instance), and a vector (shared by all the instances) or a number as is.
"""
milp_columns(x::Number, ::Vector{Int}, ::Int) = x
milp_columns(x::AbstractMatrix, idx::Vector{Int}, nb::Int) = check_batch_size(x, nb)[:, idx]
milp_columns(x::AbstractVector, ::Vector{Int}, ::Int) = x

# `bound_scale` and `obj_scale` hold one number per instance, when they are vectors
function milp_scale_columns(x::AbstractVector, idx::Vector{Int}, nb::Int)
    return check_batch_size(x, nb)[idx]
end
milp_scale_columns(x::Number, ::Vector{Int}, ::Int) = x

columns_quadratic(Q::AbstractMatrix, ::Vector{Int}, ::Int) = Q
columns_quadratic(S::BatchScaledMatrix, idx::Vector{Int}, nb::Int) = BatchScaledMatrix(S.Q, milp_scale_columns(S.scale, idx, nb))

"""
    batch_status_code(codes)

Status of a batch whose instances have all terminated with the status codes `codes`:
`MOI.OPTIMAL` if they are all optimal, otherwise `MOI.INFEASIBLE` if any of them is primal
infeasible, otherwise `MOI.DUAL_INFEASIBLE` (see [`set_termination_status!!`](@ref)).
"""
function batch_status_code(codes)
    optimal = status_code(MOI.OPTIMAL)
    infeasible = status_code(MOI.INFEASIBLE)
    all_optimal = batched_all(==(optimal), codes)
    any_infeasible = !batched_all(!=(infeasible), codes)
    return ifelse(all_optimal, optimal, ifelse(any_infeasible, infeasible, status_code(MOI.DUAL_INFEASIBLE)))
end

"""
    maybe_compact!(state, milp, algo, prog, must_terminate)

Called by the solve loops after every check. When at least half of a host batch of at least
[`COMPACTION_MIN_BATCH`](@ref) instances is frozen, solve the instances still running as a smaller
batch, write them back into `state` and return `true`; otherwise return `must_terminate`.

Only a host `Bool` stop flag and `Array` iterates qualify, so a compiled Reactant solve and the
device backends are untouched. Disabled by `batch_compaction = false`.
"""
maybe_compact!(::AbstractState, ::MILP, ::Algorithm, prog, must_terminate) = must_terminate
function maybe_compact!(state::AbstractState, milp::MILP, algo::Algorithm, prog, must_terminate::Bool)
    (must_terminate || !algo.generic.batch_compaction || within_compile()) && return must_terminate
    return compact_and_finish!(state, state.sol.x, state.stats.active, milp, algo, prog)
end

compact_and_finish!(::AbstractState, ::AbstractVecOrMat, ::Any, ::MILP, ::Algorithm, prog) = false
function compact_and_finish!(
        state::AbstractState, ::Matrix, active::Vector{Bool}, milp::MILP, algo::Algorithm, prog,
    )
    nb = length(active)
    nactive = count(active)
    (nb >= COMPACTION_MIN_BATCH && 2 * nactive <= nb && nactive > 0) || return false
    idx = findall(active)
    sub = columns(state, idx, nb)
    sub_milp = columns(milp, idx)
    must_terminate = false
    while !must_terminate
        must_terminate = check_pass!(sub, sub_milp, algo, prog)
        must_terminate = maybe_compact!(sub, sub_milp, algo, prog, must_terminate)
    end
    scatter_columns(state, sub, idx, nb)
    # the status of the sub-batch ignores the instances frozen before the compaction
    stats = state.stats
    running = status_code(MOI.OPTIMIZE_NOT_CALLED)
    if batched_all(!=(running), stats.instance_status_code)
        stats.termination_status_code = batch_status_code(stats.instance_status_code)
    end
    return true
end
