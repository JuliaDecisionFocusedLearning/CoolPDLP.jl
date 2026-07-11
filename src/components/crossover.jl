"""
    CrossoverParameters

Post-solve crossover (V1): after `TerminationStatus.OPTIMAL`, snap primal coordinates
near finite box bounds and optionally tighten bounds implied by equality rows.

This is **not** a basic-vertex crossover (no simplex basis / Megiddo pivots). It is a
lightweight rounding step related to PDLP post-processing and MIP-ready primals; see
issue #13 and the draft note in the PR. A full Megiddo-style crossover is planned
separately.

See [`crossover_threshold!`](@ref) and [`apply_crossover!`](@ref).

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct CrossoverParameters{T <: Number}
    "whether to apply crossover after PDLP/PDHG terminates optimally"
    enabled::Bool = true
    "distance to a bound below which the primal is snapped to that bound"
    threshold::T = 1.0e-6
    "tolerance for treating lower and upper bounds as equal (fixed variable)"
    fixed_tol::T = 1.0e-8
    "revert to the pre-crossover primal if KKT errors regress beyond tolerances"
    rollback_on_kkt_regression::Bool = true
    "relative KKT increase tolerated above the pre-crossover value (0 = no increase)"
    kkt_rtol::T = 0.0
    "tighten infinite bounds from equality rows before snapping"
    use_effective_bounds::Bool = true
    "tolerance for treating a coordinate as on a finite box bound"
    bound_atol::T = 1.0e-12
    "tolerance for treating a constraint row as an equality"
    eq_atol::T = 1.0e-12
end

function Base.show(io::IO, params::CrossoverParameters)
    (; enabled, threshold, fixed_tol, rollback_on_kkt_regression, kkt_rtol, use_effective_bounds, bound_atol, eq_atol) =
        params
    return print(
        io,
        "CrossoverParameters: enabled=$enabled, threshold=$threshold, fixed_tol=$fixed_tol, ",
        "rollback_on_kkt_regression=$rollback_on_kkt_regression, kkt_rtol=$kkt_rtol, ",
        "use_effective_bounds=$use_effective_bounds, bound_atol=$bound_atol, eq_atol=$eq_atol",
    )
end

"""
    crossover_kkt_acceptable(err_before, err_after, termination_reltol, params)

Return `true` if the post-crossover KKT errors should be kept.

When `rollback_on_kkt_regression` is true, reject the crossover if either:
- `relative(err_after) > termination_reltol`, or
- `relative(err_after) > relative(err_before) * (1 + kkt_rtol)`.
"""
function crossover_kkt_acceptable(
        err_before::KKTErrors,
        err_after::KKTErrors,
        termination_reltol,
        params::CrossoverParameters,
    )
    params.rollback_on_kkt_regression || return true
    rel_before = relative(err_before)
    rel_after = relative(err_after)
    rel_after <= termination_reltol || return false
    rel_after <= rel_before * (1 + params.kkt_rtol) || return false
    return true
end

function _crossover_at_box_mask(
        x::AbstractVector,
        lv::AbstractVector,
        uv::AbstractVector;
        atol::Real = 1.0e-12,
    )
    at_l = isfinite.(lv) .& (abs.(x .- lv) .<= atol)
    at_u = isfinite.(uv) .& (abs.(x .- uv) .<= atol)
    return at_l .| at_u
end

function _crossover_cpu_milp(milp::MILP)
    # GPU / CSR MILPs: run implied-bounds logic on CPU CSC (row access via `At` columns).
    milp_csc = set_matrix_type(SparseMatrixCSC, milp)
    return adapt(CPU(), milp_csc)
end

"""
    crossover_effective_bounds(milp, x)

Box bounds tightened with implied limits from equality rows.

When an equality row has exactly one variable not yet on a finite box bound, that
row's implied bound is used to fill in an infinite bound (e.g. `x₁ ≤ 1` from
`x₁ + x₂ = 1` when `x₂` is already on its lower bound).

Computed on CPU and copied back to the device of `milp.lv` / `milp.uv`.
"""
function crossover_effective_bounds(
        milp::MILP{T},
        x::AbstractVector{T},
        params::CrossoverParameters{T},
    ) where {T}
    (; bound_atol, eq_atol) = params
    lv_eff = copy(milp.lv)
    uv_eff = copy(milp.uv)
    milp_cpu = _crossover_cpu_milp(milp)
    x_cpu = Vector(x)
    lv_cpu = Vector(lv_eff)
    uv_cpu = Vector(uv_eff)
    at_box = Vector(
        _crossover_at_box_mask(x_cpu, milp_cpu.lv, milp_cpu.uv; atol = bound_atol),
    )
    _crossover_effective_bounds!(
        lv_cpu,
        uv_cpu,
        milp_cpu.At,
        milp_cpu.lc,
        milp_cpu.uc,
        x_cpu,
        at_box;
        eq_atol,
    )
    backend = get_backend(lv_eff)
    copyto!(lv_eff, adapt(backend, lv_cpu))
    copyto!(uv_eff, adapt(backend, uv_cpu))
    return lv_eff, uv_eff
end

function crossover_effective_bounds(
        milp::MILP{T},
        x::AbstractVector{T};
        bound_atol::Real = 1.0e-12,
        eq_atol::Real = 1.0e-12,
    ) where {T}
    return crossover_effective_bounds(
        milp,
        x,
        CrossoverParameters{T}(; bound_atol = T(bound_atol), eq_atol = T(eq_atol)),
    )
end

function _crossover_effective_bounds!(
        lv_eff,
        uv_eff,
        At::SparseMatrixCSC{T},
        lc,
        uc,
        x,
        at_box;
        eq_atol::Real = 1.0e-12,
    ) where {T}
    m = size(At, 2)
    lc_cpu = Vector(lc)
    uc_cpu = Vector(uc)
    x_cpu = Vector(x)
    lv_cpu = Vector(lv_eff)
    uv_cpu = Vector(uv_eff)
    at_box_cpu = Vector(at_box)
    for i in 1:m
        isapprox(lc_cpu[i], uc_cpu[i]; atol = eq_atol) || continue
        slack = lc_cpu[i]
        free_j = 0
        free_aij = zero(T)
        n_free = 0
        for ptr in nzrange(At, i)
            j = SparseArrays.rowvals(At)[ptr]
            aij = nonzeros(At)[ptr]
            if at_box_cpu[j]
                slack -= aij * x_cpu[j]
            else
                n_free += 1
                n_free > 1 && break
                free_j = j
                free_aij = aij
            end
        end
        n_free == 1 || continue
        if free_aij > 0
            uv_cpu[free_j] = min(uv_cpu[free_j], slack / free_aij)
        elseif free_aij < 0
            lv_cpu[free_j] = max(lv_cpu[free_j], slack / free_aij)
        end
    end
    lv_eff .= lv_cpu
    uv_eff .= uv_cpu
    return lv_eff, uv_eff
end

"""
    crossover_threshold!(x, lv, uv, params::CrossoverParameters)

Snap primal `x` to variable bounds using a fixed threshold.

For each coordinate: fixed variables are set to their bound; otherwise, if `x` is
within `threshold` of a finite lower or upper bound, it is moved to that bound.
"""
function crossover_threshold!(
        x::AbstractVector{T},
        lv::AbstractVector{T},
        uv::AbstractVector{T},
        params::CrossoverParameters{T},
    ) where {T}
    (; threshold, fixed_tol) = params
    fixed = abs.(lv .- uv) .<= fixed_tol
    @. x = ifelse(fixed, lv, x)
    near_l = isfinite.(lv) .& (x .- lv .<= threshold)
    @. x = ifelse(near_l, lv, x)
    near_u = isfinite.(uv) .& (uv .- x .<= threshold)
    @. x = ifelse(near_u, uv, x)
    return x
end

function crossover_threshold!(
        x::AbstractVector{T},
        milp::MILP{T},
        params::CrossoverParameters{T},
    ) where {T}
    if params.use_effective_bounds
        lv_eff, uv_eff = crossover_effective_bounds(milp, x, params)
    else
        lv_eff, uv_eff = milp.lv, milp.uv
    end
    crossover_threshold!(x, lv_eff, uv_eff, params)
    return x
end

function crossover_threshold!(
        sol::PrimalDualSolution{T},
        milp::MILP{T},
        params::CrossoverParameters{T},
    ) where {T}
    crossover_threshold!(sol.x, milp, params)
    return sol
end

function crossover_n_changed(x_after, x_before)
    if get_backend(x_after) === CPU()
        n = 0
        for i in eachindex(x_after, x_before)
            x_after[i] != x_before[i] && (n += 1)
        end
        return n
    end
    return sum(x_after .!= x_before)
end

"""
    fraction_at_bounds(x, milp; atol=1e-12)

Fraction of coordinates equal to a finite bound.
"""
function fraction_at_bounds(
        x::AbstractVector,
        milp::MILP;
        atol::Real = 1.0e-12,
    )
    (; lv, uv) = milp
    n = length(x)
    n == 0 && return 0.0
    at_l = isfinite.(lv) .& (abs.(x .- lv) .<= atol)
    at_u = isfinite.(uv) .& (abs.(x .- uv) .<= atol)
    return sum(at_l .| at_u) / n
end
