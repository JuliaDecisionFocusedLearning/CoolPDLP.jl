"""
    CrossoverParameters

Post-solve crossover settings: threshold snapping to bounds after `TerminationStatus.OPTIMAL` termination.

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
end

function Base.show(io::IO, params::CrossoverParameters)
    (; enabled, threshold, fixed_tol, rollback_on_kkt_regression, kkt_rtol, use_effective_bounds) = params
    return print(
        io,
        "CrossoverParameters: enabled=$enabled, threshold=$threshold, fixed_tol=$fixed_tol, ",
        "rollback_on_kkt_regression=$rollback_on_kkt_regression, kkt_rtol=$kkt_rtol, ",
        "use_effective_bounds=$use_effective_bounds",
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
        x::AbstractVector{T};
        bound_atol::Real = 1.0e-12,
        eq_atol::Real = 1.0e-12,
    ) where {T}
    lv_eff = copy(milp.lv)
    uv_eff = copy(milp.uv)
    milp_cpu = _crossover_cpu_milp(milp)
    x_cpu = Vector(x)
    lv_cpu = Vector(lv_eff)
    uv_cpu = Vector(uv_eff)
    at_box = Vector(
        _crossover_at_box_mask(x_cpu, Vector(milp_cpu.lv), Vector(milp_cpu.uv); atol = bound_atol),
    )
    _crossover_effective_bounds!(
        lv_cpu,
        uv_cpu,
        milp_cpu.A,
        Vector(milp_cpu.lc),
        Vector(milp_cpu.uc),
        x_cpu,
        at_box;
        eq_atol,
    )
    backend = get_backend(lv_eff)
    copyto!(lv_eff, adapt(backend, lv_cpu))
    copyto!(uv_eff, adapt(backend, uv_cpu))
    return lv_eff, uv_eff
end

function _crossover_effective_bounds!(
        lv_eff,
        uv_eff,
        A::SparseMatrixCSC{T},
        lc,
        uc,
        x,
        at_box;
        eq_atol::Real = 1.0e-12,
    ) where {T}
    m, n = size(A)
    lc_cpu = Vector(lc)
    uc_cpu = Vector(uc)
    x_cpu = Vector(x)
    lv_cpu = Vector(lv_eff)
    uv_cpu = Vector(uv_eff)
    at_box_cpu = Vector(at_box)
    for i in 1:m
        isapprox(lc_cpu[i], uc_cpu[i]; atol = eq_atol) || continue
        free = Int[]
        slack = lc_cpu[i]
        @inbounds for j in 1:n
            aij = A[i, j]
            aij == 0 && continue
            if at_box_cpu[j]
                slack -= aij * x_cpu[j]
            else
                push!(free, j)
            end
        end
        length(free) == 1 || continue
        j = only(free)
        aij = A[i, j]
        if aij > 0
            implied = slack / aij
            if !isfinite(uv_cpu[j])
                uv_cpu[j] = implied
            else
                uv_cpu[j] = min(uv_cpu[j], implied)
            end
        elseif aij < 0
            implied = slack / aij
            if !isfinite(lv_cpu[j])
                lv_cpu[j] = implied
            else
                lv_cpu[j] = max(lv_cpu[j], implied)
            end
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
        lv_eff, uv_eff = crossover_effective_bounds(milp, x)
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
