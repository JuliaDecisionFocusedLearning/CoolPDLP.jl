"""
    PDHGParameters

# Fields

$(TYPEDFIELDS)
"""
struct PDHGParameters{
        T <: Number,
        G <: GenericParameters{T},
        P <: PreconditioningParameters{T},
        S <: StepSizeParameters{T},
        F <: TerminationParameters{T},
    }
    generic::G
    preconditioning::P
    step_size::S
    termination::F

    function PDHGParameters(
            _T::Type{T} = Float64,
            ::Type{Ti} = Int,
            ::Type{M} = SparseMatrixCSC,
            backend::B = CPU();
            check_every = 100,
            record_error_history = true,
            chambolle_pock_alpha = _T(1.0),
            invnorm_scaling = _T(0.9),
            termination_reltol = _T(1.0e-4),
            max_kkt_passes = 100_000,
            time_limit = 100.0,
        ) where {T, Ti, M, B}

        generic = GenericParameters(
            T, Ti, M, backend;
            zero_tol = _T(NaN), check_every, record_error_history
        )
        preconditioning = PreconditioningParameters(;
            chambolle_pock_alpha = _T(chambolle_pock_alpha), ruiz_iter = 0
        )
        step_size = StepSizeParameters(;
            invnorm_scaling = _T(invnorm_scaling)
        )
        termination = TerminationParameters(;
            termination_reltol = _T(termination_reltol), max_kkt_passes, time_limit
        )

        return new{T, typeof(generic), typeof(preconditioning), typeof(step_size), typeof(termination)}(
            generic,
            preconditioning,
            step_size,
            termination
        )
    end
end

@kwdef struct PDHGScratch{T <: Number, V <: AbstractVector{T}}
    x::V
    y::V
    r::V
end


"""
    PDHGState

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDHGState{
        T <: Number, V <: AbstractVector{T},
    }
    "current primal solution"
    x::V
    "current dual solution"
    y::V
    "step size"
    η::T
    "primal weight"
    ω::T
    "scratch space"
    scratch::PDHGScratch{T, V}
    "scales of the feasibility errors"
    scales::FeasibilityErrorScales{T}
    "convergence stats"
    stats::ConvergenceStats{T}
end

"""
    pdhg(
        milp::MILP,
        params::PDHGParameters,
        x_init=zero(milp.lv),
        y_init=zero(milp.lc);
        show_progress::Bool=true
    )
    
Apply the PDHG algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init` and dual variable `y_init`.

Return a couple `(x, y), stats` where `x` is the primal solution, `y` is the dual solution and `stats` contains convergence information.
"""
function pdhg(
        milp_init::MILP,
        params::PDHGParameters,
        x_init::AbstractVector = zero(milp_init.lv),
        y_init::AbstractVector = zero(milp_init.lc);
        show_progress::Bool = true,
    )
    starting_time = time()
    milp, x, y = preprocess(milp_init, x_init, y_init, params)
    state = initialize(milp, x, y, params; starting_time)
    pdhg!(state, milp, params; show_progress)
    return get_solution(state, milp), state.stats
end

function preprocess(
        milp_init::MILP,
        x_init::AbstractVector,
        y_init::AbstractVector,
        params::PDHGParameters,
    )
    p = pdlp_preconditioner(milp_init, params.preconditioning)
    milp_precond, x_precond, y_precond = precondition(milp_init, x_init, y_init, p)
    milp, x, y = to_device(milp_precond, x_precond, y_precond, params.generic)
    return milp, x, y
end

function initialize(
        milp::MILP{T},
        x::AbstractVector,
        y::AbstractVector,
        params::PDHGParameters;
        starting_time::Float64
    ) where {T}
    η = fixed_stepsize(milp, params.step_size)
    ω = one(η)
    scratch = PDHGScratch(; x = similar(x), y = similar(y), r = similar(x))
    scales = FeasibilityErrorScales(milp)
    stats = ConvergenceStats(T; starting_time)
    state = PDHGState(; x, y, η, ω, scratch, scales, stats)
    return state
end

function pdhg!(
        state::PDHGState,
        milp::MILP,
        params::PDHGParameters;
        show_progress::Bool,
    )
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.generic.check_every
            step!(state, milp)
            next!(prog; showvalues = (("relative_error", relative(state.stats.err)),))
        end
        termination_check!(state, milp, params)
        if !isnothing(state.stats.termination_status)
            break
        end
    end
    finish!(prog)
    return state
end

function step!(
        state::PDHGState{T, V},
        milp::MILP{T, V},
    ) where {T, V}
    (; x, y, η, ω, scratch) = state
    (; c, lv, uv, A, At, lc, uc) = milp

    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - At * y))
    At_y = mul!(scratch.x, At, y)
    xdiff = @. scratch.x = 2 * proj_box(x - τ * (c - At_y), lv, uv) - x

    # yp = y - σ * A * (2xp - x) - σ proj_{-Y}(y / σ - A * (2xp - x)))
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. y = y - σ * A_xdiff - σ * proj_box(inv(σ) * y - A_xdiff, -uc, -lc)

    @. x = (xdiff + x) / 2  # TODO: ditch this one

    state.stats.kkt_passes += 1
    return nothing
end

function kkt_errors!(
        state::PDHGState,
        milp::MILP{T, V},
    ) where {T, V}
    # TODO: go back to initial problem
    (; x, y, scratch, scales) = state
    (; c, lv, uv, A, At, lc, uc) = milp

    A_x = mul!(scratch.y, A, x)
    At_y = mul!(scratch.x, At, y)
    r = @. scratch.r = proj_multiplier(c - At_y, lv, uv)

    pc = pm(y, lc, uc)
    pv = pm(r, lv, uv)

    gap = abs(dot(c, x) + pc + pv)
    gap_scale = one(T) + abs(pc + pv) + abs(dot(c, x))

    primal_diff = @. scratch.y = A_x - proj_box(A_x, lc, uc)
    primal = norm(primal_diff)

    dual_diff = @. scratch.x = c - At_y - r
    dual = norm(dual_diff)

    return KKTErrors(;
        primal,
        dual,
        gap,
        primal_scale = scales.primal,
        dual_scale = scales.dual,
        gap_scale,
    )
end

function termination_check!(
        state::PDHGState,
        milp::MILP,
        params::PDHGParameters
    )
    (; stats) = state
    stats.time_elapsed = time() - stats.starting_time
    stats.err = kkt_errors!(state, milp)
    if params.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, stats.err))
    end
    stats.termination_status = termination_status(stats, params.termination)
    return nothing
end

function get_solution(state::PDHGState, milp::MILP)
    (; x, y) = state
    (; D1, D2) = milp
    return unprecondition_variables(x, y, Preconditioner(D1, D2))
end
