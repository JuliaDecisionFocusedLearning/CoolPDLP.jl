@kwdef struct FeasibilityErrorScales{T <: Number}
    primal::T
    dual::T
end

function FeasibilityErrorScales(milp::MILP{T}) where {T}
    (; lc, uc, c) = milp
    primal = one(T) + norm(bound_scale.(lc, uc))
    dual = one(T) + norm(c)
    return FeasibilityErrorScales(; primal, dual)
end

"""
    KKTErrors

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct KKTErrors{T <: Number}
    "primal feasibility error"
    primal::T
    primal_scale::T
    "dual feasibility error"
    dual::T
    dual_scale::T
    "primal-dual gap error"
    gap::T
    gap_scale::T
end

function Base.show(io::IO, err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return print(
        io, """KKT errors:
        - primal $(primal) (scale $(primal_scale))
        - dual $(dual) (scale $(dual_scale))
        - gap $(gap) (scale $(gap_scale))"""
    )
end

function KKTErrors(::Type{T}) where {T}
    return KKTErrors(
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
        convert(T, NaN),
    )
end


function relative(err::KKTErrors)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    return max(primal / primal_scale, dual / dual_scale, gap / gap_scale)
end
