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

    return AlgorithmParameters(
        :PDHG,
        generic,
        preconditioning,
        step_size,
        termination
    )
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
    "convergence stats"
    stats::ConvergenceStats{T}
end

"""
    pdhg(
        milp::MILP,
        params::AlgorithmParameters,
        x_init=zero(milp.lv),
        y_init=zero(milp.lc);
        show_progress::Bool=true
    )
    
Apply the PDHG algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init` and dual variable `y_init`.

Return a couple `(x, y), stats` where `x` is the primal solution, `y` is the dual solution and `stats` contains convergence information.
"""
function pdhg(
        milp_init_cpu::MILP,
        params::AlgorithmParameters,
        x_init_cpu::AbstractVector = zero(milp_init_cpu.lv),
        y_init_cpu::AbstractVector = zero(milp_init_cpu.lc);
        show_progress::Bool = true,
    )
    starting_time = time()
    milp, x, y = preprocess(milp_init_cpu, x_init_cpu, y_init_cpu, params)
    milp_init = to_device(milp_init_cpu, params.generic)
    state = initialize(milp, x, y, params; starting_time)
    pdhg!(state, milp, milp_init, params; show_progress)
    return get_solution(state, milp), state, milp
end

function preprocess(
        milp_init_cpu::MILP,
        x_init_cpu::AbstractVector,
        y_init_cpu::AbstractVector,
        params::AlgorithmParameters,
    )
    # on CPU
    p = pdlp_preconditioner(milp_init_cpu, params.preconditioning)
    milp = precondition_problem(milp_init_cpu, p)
    x, y = precondition_variables(x_init_cpu, y_init_cpu, p)

    # moving to GPU
    milp = to_device(milp, params.generic)
    x = to_device(x, params.generic)
    y = to_device(y, params.generic)

    return milp, x, y
end

function initialize(
        milp::MILP{T},
        x::AbstractVector,
        y::AbstractVector,
        params::AlgorithmParameters;
        starting_time::Float64
    ) where {T}
    η = fixed_stepsize(milp, params.step_size)
    ω = one(η)
    scratch = PDHGScratch(; x = similar(x), y = similar(y), r = similar(x))
    stats = ConvergenceStats(T; starting_time)
    state = PDHGState(; x, y, η, ω, scratch, stats)
    return state
end

function pdhg!(
        state::PDHGState,
        milp::MILP,
        milp_init::MILP,
        params::AlgorithmParameters;
        show_progress::Bool,
    )
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.generic.check_every
            step!(state, milp)
            next!(prog; showvalues = (("relative_error", relative(state.stats.err)),))
        end
        termination_check!(state, milp, milp_init, params)
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

    # xp = proj_box.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.x, At, y)
    xdiff = @. scratch.x = 2 * proj_box(x - τ * (c - At_y), lv, uv) - x

    # yp = y - σ * A * (2xp - x) - σ * proj_box.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. y = y - σ * A_xdiff - σ * proj_box(inv(σ) * y - A_xdiff, -uc, -lc)
    @. x = (xdiff + x) / 2  # TODO: ditch this one

    state.stats.kkt_passes += 1
    return nothing
end

function kkt_errors!(
        state::PDHGState,
        milp::MILP{T},
        milp_init::MILP
    ) where {T}
    # TODO: go back to initial problem
    (; scratch) = state
    prec = Preconditioner(milp.D1, milp.D2)
    x, y = precondition_variables(state.x, state.y, inv(prec))
    (; c, lv, uv, A, At, lc, uc) = milp_init

    A_x = mul!(scratch.y, A, x)
    At_y = mul!(scratch.x, At, y)
    r = @. scratch.r = proj_multiplier(c - At_y, lv, uv)

    primal_diff = @. scratch.y = A_x - proj_box(A_x, lc, uc)
    primal = norm(primal_diff)
    primal_scale = one(T) + sqrt(mapreduce(squared_bound_scale, +, lc, uc))

    dual_diff = @. scratch.x = c - At_y - r
    dual = norm(dual_diff)
    dual_scale = one(T) + norm(c)

    pc = p(-y, lc, uc)
    pv = p(-r, lv, uv)

    gap = abs(dot(c, x) + pc + pv)
    gap_scale = one(T) + abs(pc + pv) + abs(dot(c, x))

    err = KKTErrors(;
        primal,
        dual,
        gap,
        primal_scale,
        dual_scale,
        gap_scale,
    )
    return err
end

function termination_check!(
        state::PDHGState,
        milp::MILP,
        milp_init::MILP,
        params::AlgorithmParameters
    )
    (; stats) = state
    stats.time_elapsed = time() - stats.starting_time
    stats.err = kkt_errors!(state, milp, milp_init)
    if params.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, stats.err))
    end
    stats.termination_status = termination_status(stats, params.termination)
    return nothing
end

function get_solution(state::PDHGState, milp::MILP)
    (; x, y) = state
    (; D1, D2) = milp
    return precondition_variables(x, y, inv(Preconditioner(D1, D2)))
end
