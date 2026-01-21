module CoolPDLPMOIExt

import MathOptInterface as MOI
import CoolPDLP
import SparseArrays: SparseMatrixCSC

function __init__()
    return Base.setglobal!(CoolPDLP, :Optimizer, Optimizer)
end

MOI.Utilities.@product_of_sets(
    RHS,
    MOI.EqualTo{T},
    MOI.GreaterThan{T},
    MOI.LessThan{T},
    MOI.Interval{T},
)

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    obj_value::T
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    solve_time::T
    silent::Bool
    sets::Union{Nothing, RHS{T}}
    options::Dict{Symbol, Any}

    function Optimizer(; T = Float64)
        return new{T}(
            T[], T[], T[], zero(T),
            MOI.OPTIMIZE_NOT_CALLED, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS,
            zero(T), false, nothing, Dict{Symbol, Any}(),
        )
    end
end

function MOI.is_empty(model::Optimizer)
    return (
        isempty(model.x) &&
            isempty(model.y) &&
            isempty(model.z) &&
            iszero(model.obj_value) &&
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
    model.obj_value = zero(T)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.primal_status = MOI.UNKNOWN_RESULT_STATUS
    model.dual_status = MOI.UNKNOWN_RESULT_STATUS
    model.solve_time = zero(T)
    model.sets = nothing
    return
end

const SUPPORTED_SET_TYPE{T} = Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{<:SUPPORTED_SET_TYPE{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{<:SUPPORTED_SET_TYPE{T}}) where {T} = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where {T} = true

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

MOI.get(model::Optimizer, ::MOI.TimeLimitSec) = get(model.options, :time_limit, nothing)
function MOI.set(model::Optimizer{T}, ::MOI.TimeLimitSec, value) where {T}
    return if isnothing(value)
        delete!(model.options, :time_limit)
    else
        model.options[:time_limit] = T(value)
    end
end
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool) = (model.silent = value;)
MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute) = model.options[Symbol(attr.name)]
MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value) = (model.options[Symbol(attr.name)] = value;)

MOI.get(::Optimizer, ::MOI.SolverName) = "CoolPDLP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(CoolPDLP))
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status
MOI.get(model::Optimizer, ::MOI.ResultCount) = model.termination_status == MOI.OPTIMAL ? 1 : 0
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
    row = only(MOI.Utilities.rows(model.sets, ci))
    return model.y[row]
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{T}},
    ) where {T}
    MOI.check_result_index_bounds(model, attr)
    return max(model.z[ci.value], zero(T))
end

function MOI.get(
        model::Optimizer{T},
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{T}},
    ) where {T}
    MOI.check_result_index_bounds(model, attr)
    return min(model.z[ci.value], zero(T))
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

function MOI.optimize!(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    MOI.empty!(dest)
    cache = OptimizerCache{T}()
    index_map = MOI.copy_to(cache, src)

    n = cache.constraints.coefficients.n
    max_sense = cache.objective.sense == MOI.MAX_SENSE

    A = convert(SparseMatrixCSC{T, Int}, cache.constraints.coefficients)

    c = zeros(T, n)
    obj_constant = zero(T)
    if cache.objective.scalar_affine !== nothing
        for term in cache.objective.scalar_affine.terms
            c[term.variable.value] += term.coefficient
        end
        obj_constant = cache.objective.scalar_affine.constant
    end
    if max_sense
        c .*= -one(T)
    end

    dest.sets = cache.constraints.sets
    lv = cache.variables.lower
    uv = cache.variables.upper
    lc = cache.constraints.constants.lower
    uc = cache.constraints.constants.upper

    milp = CoolPDLP.MILP(; c, lv, uv, A, lc, uc)

    algorithm = pop!(dest.options, :algorithm, CoolPDLP.PDLP)

    float_type = pop!(dest.options, :float_type, T)
    int_type = pop!(dest.options, :int_type, Int)  # FIXME: get int type from float type?
    matrix_type = pop!(dest.options, :matrix_type, SparseMatrixCSC)

    algo_opts = Dict{Symbol, Any}(:show_progress => !dest.silent)
    for (k, v) in dest.options
        algo_opts[k] = v
    end
    algo = algorithm(float_type, int_type, matrix_type; algo_opts...)

    sol, stats = CoolPDLP.solve(milp, algo)

    dest.x = Array(sol.x)
    dest.y = Array(sol.y)
    dest.z = CoolPDLP.proj_multiplier.(c .- milp.At * dest.y, lv, uv)

    raw_obj = CoolPDLP.objective_value(dest.x, milp)
    dest.obj_value = (max_sense ? -raw_obj : raw_obj) + obj_constant
    dest.solve_time = stats.time_elapsed

    cts = stats.termination_status
    ts, ps, ds = if cts == CoolPDLP.OPTIMAL
        MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT
    elseif cts == CoolPDLP.TIME_LIMIT
        MOI.TIME_LIMIT, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS
    elseif cts == CoolPDLP.ITERATION_LIMIT
        MOI.ITERATION_LIMIT, MOI.UNKNOWN_RESULT_STATUS, MOI.UNKNOWN_RESULT_STATUS
    else
        @assert cts == CoolPDLP.STILL_RUNNING
        MOI.OTHER_ERROR, MOI.NO_SOLUTION, MOI.NO_SOLUTION
    end
    dest.termination_status = ts
    dest.primal_status = ps
    dest.dual_status = ds

    return index_map, false
end

end
