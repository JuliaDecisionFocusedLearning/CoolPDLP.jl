"""
    $(TYPEDSIGNATURES)
"""
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

    return Parameters{PDHG}(
        generic,
        preconditioning,
        step_size,
        termination
    )
end

"""
    PDHGState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{
        T <: Number, V <: DenseVector{T},
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "next solution"
    sol_next::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::StepSizes{T}
    "scratch space"
    scratch::Scratch{T, V}
    "convergence stats"
    stats::ConvergenceStats{T}
end

function initialize(
        milp::MILP{T, V},
        sol::PrimalDualSolution{T, V},
        params::Parameters{PDHG, T};
        starting_time::Float64
    ) where {T, V}
    sol_next = zero(sol)
    η = fixed_stepsize(milp, params.step_size)
    ω = one(η)
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(; x = similar(sol.x), y = similar(sol.y), r = similar(sol.x))
    stats = ConvergenceStats(T; starting_time)
    state = PDHGState(; sol, sol_next, step_sizes, scratch, stats)
    return state
end

function solve!(
        state::PDHGState,
        milp::MILP,
        params::Parameters;
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
    (; sol, sol_next, step_sizes, scratch) = state
    (; η, ω) = step_sizes
    (; c, lv, uv, A, At, lc, uc) = milp

    τ, σ = η / ω, η * ω

    # xp = proj_box.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.x, At, sol.y)
    @. sol_next.x = proj_box(sol.x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.x = 2sol_next.x - sol.x

    # yp = y - σ * A * (2xp - x) - σ * proj_box.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. sol_next.y = sol.y - σ * A_xdiff - σ * proj_box(inv(σ) * sol.y - A_xdiff, -uc, -lc)

    state.sol, state.sol_next = state.sol_next, state.sol

    state.stats.kkt_passes += 1
    return nothing
end
