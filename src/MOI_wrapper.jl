"Dual objective `lᵀ|y|⁺ - uᵀ|y|⁻ + lᵥᵀ|z|⁺ - uᵥᵀ|z|⁻ + c0` of the multipliers `y` and `z`."
function _dual_objective(y, z, milp::MILP, c0)
    (; lv, uv, lc, uc) = milp
    return (
        sum(safeprod_left.(lc, positive_part.(y)); init = zero(c0))
            - sum(safeprod_left.(uc, negative_part.(y)); init = zero(c0))
            + sum(safeprod_left.(lv, positive_part.(z)); init = zero(c0))
            - sum(safeprod_left.(uv, negative_part.(z)); init = zero(c0))
            + c0
    )
end

MOI.Utilities.@product_of_sets(
    RHS,
    MOI.EqualTo{T},
    MOI.GreaterThan{T},
    MOI.LessThan{T},
    MOI.Interval{T},
)

"""
    Optimizer

Solver type compatible with JuMP, which calls an algorithm from `CoolPDLP` under the hood.

Its options are the same as the keyword arguments of [`Algorithm`](@ref).

The objective may be affine or a convex quadratic (`MOI.ScalarQuadraticFunction`). A quadratic
objective which is detectably nonconvex (see [`check_convex`](@ref)) makes `optimize!` throw.
"""
mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    obj_value::T
    dual_obj_value::T
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    solve_time::Float64
    silent::Bool
    sets::Union{Nothing, RHS{T}}
    options::Dict{Symbol, Any}

    function Optimizer{T}() where {T <: Real}
        return new{T}(
            T[], T[], T[], T(NaN), T(NaN),
            MOI.OPTIMIZE_NOT_CALLED, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS,
            0.0, false, nothing, Dict{Symbol, Any}(),
        )
    end
end

Optimizer() = Optimizer{Float64}()

function MOI.is_empty(model::Optimizer)
    return (
        isempty(model.x) &&
            isempty(model.y) &&
            isempty(model.z) &&
            isnan(model.obj_value) &&
            isnan(model.dual_obj_value) &&
            model.termination_status == MOI.OPTIMIZE_NOT_CALLED &&
            model.primal_status == MOI.UNKNOWN_RESULT_STATUS &&
            model.dual_status == MOI.UNKNOWN_RESULT_STATUS &&
            iszero(model.solve_time) &&
            isnothing(model.sets)
    )
end
function MOI.empty!(model::Optimizer{T}) where {T}
    empty!(model.x)
    empty!(model.y)
    empty!(model.z)
    model.obj_value = T(NaN)
    model.dual_obj_value = T(NaN)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.primal_status = MOI.UNKNOWN_RESULT_STATUS
    model.dual_status = MOI.UNKNOWN_RESULT_STATUS
    model.solve_time = 0.0
    model.sets = nothing
    return
end

const SUPPORTED_SET_TYPE{T} = Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{<:SUPPORTED_SET_TYPE{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{<:SUPPORTED_SET_TYPE{T}}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}) where {T} = true

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    haskey(model.options, :time_limit) || return nothing
    # a `RawOptimizerAttribute("time_limit")` may have stored any real number
    return convert(Float64, model.options[:time_limit])
end
function MOI.set(model::Optimizer{T}, ::MOI.TimeLimitSec, value) where {T}
    if isnothing(value)
        delete!(model.options, :time_limit)
    else
        model.options[:time_limit] = Float64(value)
    end
    return nothing
end
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool) = (model.silent = value; nothing)
MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute) = model.options[Symbol(attr.name)]
MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value) = (model.options[Symbol(attr.name)] = value; nothing)

MOI.get(::Optimizer, ::MOI.SolverName) = "CoolPDLP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(CoolPDLP))
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status
MOI.get(model::Optimizer, ::MOI.ResultCount) = model.termination_status != MOI.OPTIMIZE_NOT_CALLED ? 1 : 0
MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.termination_status)
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function _status_check_index(model, attr, ret)
    if attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return ret
end
function _attr_check_index(model, attr, ret)
    MOI.check_result_index_bounds(model, attr)
    return ret
end

MOI.get(model::Optimizer, attr::MOI.PrimalStatus) = _status_check_index(model, attr, model.primal_status)
MOI.get(model::Optimizer, attr::MOI.DualStatus) = _status_check_index(model, attr, model.dual_status)
MOI.get(model::Optimizer, attr::MOI.ObjectiveValue) = _attr_check_index(model, attr, model.obj_value)
MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue) = _attr_check_index(model, attr, model.dual_obj_value)

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.x[vi.value]
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S},
    ) where {T, S <: SUPPORTED_SET_TYPE{T}}
    MOI.check_result_index_bounds(model, attr)
    return model.y[MOI.Utilities.rows(model.sets, ci)]
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{T}},
    ) where {T}
    MOI.check_result_index_bounds(model, attr)
    return positive_part(model.z[ci.value])
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{T}},
    ) where {T}
    MOI.check_result_index_bounds(model, attr)
    return -negative_part(model.z[ci.value])
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, S},
    ) where {T, S <: Union{MOI.EqualTo{T}, MOI.Interval{T}}}
    MOI.check_result_index_bounds(model, attr)
    return model.z[ci.value]
end

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing},
        MOI.Utilities.Hyperrectangle{T}, RHS{T},
    },
}

MOI.default_cache(::Optimizer{T}, ::Type{T}) where {T} = MOI.Utilities.UniversalFallback(OptimizerCache{T}())

function MOI.optimize!(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    cache = MOI.default_cache(dest, T)
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end

function _pass_attributes(dest::Optimizer{T}, cache::MOI.Utilities.UniversalFallback{OptimizerCache{T}}, index_map) where {T}
    MOI.Utilities.pass_attributes(dest, cache, index_map, MOI.get(cache, MOI.ListOfVariableIndices()))

    attrs = MOI.Utilities.ModelFilter(a -> !(a isa MOI.ObjectiveSense || a isa MOI.ObjectiveFunction), cache)
    MOI.Utilities.pass_attributes(dest, attrs, index_map)

    for (F, S) in MOI.get(cache, MOI.ListOfConstraintTypesPresent())
        idxs = MOI.get(cache, MOI.ListOfConstraintIndices{F, S}())
        MOI.Utilities.pass_attributes(dest, cache, index_map, idxs)
    end

    return cache.model
end

"""
    quadratic_objective_matrix(f::MOI.ScalarQuadraticFunction, n)

Return the symmetric matrix `Q` such that the quadratic part of `f` is `½ xᵀQx`.

MOI already uses this convention: a diagonal term with coefficient `a` stands for `½ a xᵢ²`, and
an off-diagonal one for `a xᵢ xⱼ`, so it contributes `a` to both `Qᵢⱼ` and `Qⱼᵢ`.
"""
function quadratic_objective_matrix(f::MOI.ScalarQuadraticFunction{T}, n::Integer) where {T}
    I, J, V = Int[], Int[], T[]
    for term in f.quadratic_terms
        i, j, a = term.variable_1.value, term.variable_2.value, term.coefficient
        push!(I, i); push!(J, j); push!(V, a)
        if i != j
            push!(I, j); push!(J, i); push!(V, a)
        end
    end
    return sparse(I, J, V, n, n)
end

function MOI.optimize!(dest::Optimizer{T}, fcache::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
    MOI.empty!(dest)
    index_map = MOI.Utilities.identity_index_map(fcache)
    cache = _pass_attributes(dest, fcache, index_map)

    n = cache.constraints.coefficients.n
    max_sense = cache.objective.sense == MOI.MAX_SENSE

    A = convert(SparseMatrixCSC{T, Int}, cache.constraints.coefficients)

    c = zeros(T, n)
    c0 = zero(T)
    affine = cache.objective.scalar_affine
    quadratic = cache.objective.scalar_quadratic
    if affine !== nothing
        for term in affine.terms
            c[term.variable.value] += term.coefficient
        end
        c0 = affine.constant
    end
    Q = if quadratic !== nothing
        for term in quadratic.affine_terms
            c[term.variable.value] += term.coefficient
        end
        c0 = quadratic.constant
        quadratic_objective_matrix(quadratic, n)
    else
        nothing
    end
    if max_sense  # a `MILP` is a minimization problem
        c .*= -one(T)
        c0 = -c0
        isnothing(Q) || (Q .*= -one(T))
    end
    isnothing(Q) || check_convex(Q)

    dest.sets = cache.constraints.sets
    lv = cache.variables.lower
    uv = cache.variables.upper
    lc = cache.constraints.constants.lower
    uc = cache.constraints.constants.upper

    milp = isnothing(Q) ? MILP(; c, c0, lv, uv, A, lc, uc) : MILP(; c, c0, lv, uv, A, Q, lc, uc)

    algorithm = get(dest.options, :algorithm, PDLP)

    float_type = get(dest.options, :float_type, T)
    if float_type !== T
        @warn "Got mismatched float type: solving in $float_type but returning the solution in $T."
    end
    int_type = get(dest.options, :int_type, Int)
    matrix_type = get(dest.options, :matrix_type, SparseMatrixCSC)

    # these four keys configure `optimize!` itself rather than `Algorithm`, so they must stay in
    # `dest.options` (unlike `pop!`) for a second `optimize!` call or a later `MOI.get` to see them
    algorithm_option_keys = (:algorithm, :float_type, :int_type, :matrix_type)
    algo_opts = Dict{Symbol, Any}(:show_progress => !dest.silent)
    for (k, v) in dest.options
        k in algorithm_option_keys && continue
        algo_opts[k] = k === :termination_norm && v isa AbstractString ? termination_norm(v) : v
    end
    algo = algorithm(float_type, int_type, matrix_type; algo_opts...)

    # `algo` comes from untyped options, so it is inferred as `Any`: with a direct call, the
    # compiler would still infer the only matching `solve` method on the abstract `Algorithm`,
    # which takes many minutes (the whole solver over abstract types). `invokelatest` keeps that
    # call opaque, and `solve` gets compiled at run time for the concrete algorithm instead.
    sol, stats = Base.invokelatest(solve, milp, algo)

    cts = termination_status(stats)
    @assert cts != MOI.OPTIMIZE_NOT_CALLED "solve did not reach a terminal status"
    dest.termination_status = cts
    dest.solve_time = stats.time_elapsed

    if cts == MOI.INFEASIBLE
        # the dual ray `y` and its reduced cost `-Aᵀy` certify primal infeasibility, the ray
        # objective (without the objective constant) is positive
        dest.x = fill(T(NaN), n)
        # no ray at all when the bounds cross (or the detection is disabled)
        y = isnothing(stats.ray) ? zeros(T, size(A, 1)) : Array(stats.ray.y)
        scale = maximum(abs, y; init = zero(eltype(y)))
        dest.y = iszero(scale) ? y : y ./ scale
        dest.z = proj_multiplier.(-(milp.At * dest.y), lv, uv)
        raw_dual_obj = _dual_objective(dest.y, dest.z, milp, zero(T))
        dest.obj_value = T(NaN)
        dest.dual_obj_value = max_sense ? -raw_dual_obj : raw_dual_obj
        has_ray = !iszero(scale) && isfinite(raw_dual_obj) && raw_dual_obj > 0
        dest.primal_status = MOI.NO_SOLUTION
        dest.dual_status = has_ray ? MOI.INFEASIBILITY_CERTIFICATE : MOI.UNKNOWN_RESULT_STATUS
    elseif cts == MOI.DUAL_INFEASIBLE
        # the primal ray `x` certifies dual infeasibility, its objective (without the objective
        # constant) is negative in the minimization form; for a QP this only checks `cᵀd < 0`,
        # since the detector already required `Q d ≈ 0` (see `primal_ray_error!!`)
        x = Array(stats.ray.x)
        scale = maximum(abs, x; init = zero(eltype(x)))
        dest.x = iszero(scale) ? x : x ./ scale
        dest.y = fill(T(NaN), size(A, 1))
        dest.z = fill(T(NaN), n)
        raw_obj = dot(c, dest.x)
        dest.obj_value = max_sense ? -raw_obj : raw_obj
        dest.dual_obj_value = T(NaN)
        dest.primal_status = raw_obj < 0 ? MOI.INFEASIBILITY_CERTIFICATE : MOI.UNKNOWN_RESULT_STATUS
        dest.dual_status = MOI.NO_SOLUTION
    else
        dest.x = Array(sol.x)
        dest.y = Array(sol.y)
        Qx = isnothing(Q) ? zero(c) : Q * dest.x
        dest.z = proj_multiplier.(c .+ Qx .- milp.At * dest.y, lv, uv)
        raw_obj = objective_value(dest.x, milp)
        # the dual objective of a QP also loses ½xᵀQx
        raw_dual_obj = _dual_objective(dest.y, dest.z, milp, c0) - dot(dest.x, Qx) / 2
        dest.obj_value = max_sense ? -raw_obj : raw_obj
        dest.dual_obj_value = max_sense ? -raw_dual_obj : raw_dual_obj
        dest.primal_status, dest.dual_status = if cts == MOI.OPTIMAL
            MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT
        else
            MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS
        end
    end

    return index_map, false
end

function termination_norm(name::AbstractString)
    name == "L2" && return L2Norm()
    name == "Linf" && return LInfNorm()
    throw(ArgumentError("termination_norm must be \"L2\", \"Linf\" or a `TerminationNorm`, got $(repr(name))"))
end
