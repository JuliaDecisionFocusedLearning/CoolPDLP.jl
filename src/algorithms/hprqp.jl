"""
    HPRQP(args...; kwargs...)

Shortcut for [`Algorithm{:HPRQP}`](@ref): restarted Halpern Peaceman-Rachford method for convex
quadratic programs, after [HPR-QP](https://github.com/PolyU-IOR/HPR-QP) (Chen, Sun, Yuan, Zhang
and Zhao, 2025).

Unlike [`HPDLP`](@ref), which linearizes the objective at the current iterate and must shrink its
step size by `‖Q‖`, the objective is linearized at an auxiliary variable `w` with the proximal
term `σ (λ_Q I - Q)`, so the steps only depend on `‖A‖`, and restarts measure the fixed-point
residual in the metric of that splitting (see [`hprqp_fixed_point_error!!`](@ref)).

The primal step `σ = η / ω` (with `η = invnorm_scaling / ‖A‖` fixed) is set as in HPR-QP:
`‖b‖ / ‖c‖` at the start, then at each restart by a golden-section search on HPR-QP's model of
the convergence rate, see [`hprqp_sigma_update!`](@ref). Restarts follow HPR-QP's rules, see
[`hprqp_restart_check!`](@ref). A diagonal `Q` is handled exactly, with `λ_Q I` replaced by
`Diagonal(Q)` in the proximal term, on every sparse format (a dense `Q` uses `λ_Q I`).

Its defaults follow HPR-QP: eigenvalue bounds `1.05 ‖A‖²` and `1.05 ‖Q‖` (`invnorm_scaling =
1/√1.05`), decays `0.2` (sufficient) and `0.8` (necessary), a check every `100` iterations, and
Curtis-Reid, Ruiz and Pock-Chambolle scaling. Its stopping rule is also HPR-QP's: the relative
KKT errors default to `relative_kkt = Val(:pdqp)` (infinity norms and max-type scales, see
[`kkt_errors!`](@ref)); pass `relative_kkt = Val(:pdlp)` for the PDLP definition.
"""
function HPRQP(args...; kwargs...)
    return Algorithm{:HPRQP}(
        args...;
        invnorm_scaling = 1 / sqrt(1.05),
        sufficient_decay = 0.2,
        necessary_decay = 0.8,
        check_every = 100,
        curtis_reid_iter = 10,
        relative_kkt = Val(:pdqp),
        kwargs...
    )
end

supports_quadratic(::Algorithm{:HPRQP}) = true

"""
    HPRQPState

The Halpern iterate is `z = (x, y, w, Aᵀy)`: carrying `Aᵀy` along saves a product by `Aᵀ` at
each step, as in HPR-QP.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct HPRQPState{
        T <: Number,
        V <: AbstractVecOrMat{T},
        SS <: StepSizes,
        Sc <: Scratch,
        HS <: HalpernStats,
        CS <: ConvergenceStats,
        IW <: Union{Nothing, InfeasibilityWorkspace},
        N <: Number,
        B <: BatchedNumber,
        W <: AbstractVector{T},
    } <: AbstractState{T, V}
    "primal-dual part `(x̄, ȳ)` of the image `T(z)` of the last step, which is the reported solution"
    sol::PrimalDualSolution{T, V}
    "primal-dual part of the Halpern iterate"
    z::PrimalDualSolution{T, V}
    "primal-dual part of the Halpern anchor"
    z_anchor::PrimalDualSolution{T, V}
    "auxiliary variable `w` of the Halpern iterate, at which the objective is linearized"
    w::V
    "image `w̄` of `w` by the last step"
    w_bar::V
    "auxiliary variable of the anchor"
    w_anchor::V
    "`Aᵀy` of the Halpern iterate"
    aty::V
    "`Aᵀȳ` of the image of the last step"
    aty_bar::V
    "`Aᵀy` of the anchor"
    aty_anchor::V
    "primal scratch for the fixed-point residual"
    buf1::V
    "primal scratch for the fixed-point residual"
    buf2::V
    "upper bound `λ_A` on the largest eigenvalue of `AAᵀ`"
    λ_A::N
    "upper bound `λ_Q` on the largest eigenvalue of `Q`"
    λ_Q::N
    "diagonal of the proximal operator `Λ ⪰ Q` of the `w` step: `diag(Q)` for a diagonal `Q`, `λ_Q` everywhere otherwise"
    λ_Qv::W
    "whether `λ_Qv` is `diag(Q)` (a diagonal `Q`), which selects HPR-QP's per-coordinate σ model"
    Q_diagonal::Bool
    "whether more than 80% of the variables are free, which selects HPR-QP's \"noC\" splitting (no `Aᵀ` correction of `w`)"
    free_splitting::Bool
    "per-column scratch of the golden-section search of the per-coordinate σ model"
    golden::NTuple{10, B}
    "fixed-point residual of the first check, which decides the length of the long inner loops of HPR-QP"
    fpe_reference::B
    "per-column scratch"
    t1::B
    "per-column scratch"
    t2::B
    "per-column scratch"
    t3::B
    "primal scratch for the σ update"
    buf3::V
    "primal scratch for the σ update"
    buf4::V
    "step sizes, where `η / ω` is the primal step `σ` of HPR-QP"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "Halpern and restart bookkeeping"
    halpern::HS
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
end

# `λ_Qv` holds the eigenvalues of the shared `Q`, so it is not split by instance when a batch is
# compacted (`compaction_shared_fields` gets its fallback from the batch compaction code)
function compaction_shared_fields end
compaction_shared_fields(::HPRQPState) = (:λ_Qv,)

nbinstances((; sol)::HPRQPState) = nbinstances(sol)

"See [`ray_candidates`](@ref) for HPDLP."
ray_candidates(state::HPRQPState) = ((state.sol, state.z_anchor), (state.sol, nothing))

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:HPRQP};
        starting_time::Float64
    ) where {T, V}
    (; A, At, Q) = milp
    params = algo.step_size
    (; invnorm_scaling) = params
    norm_A = spectral_norm(A, At; tol = params.spectral_norm_tol, maxiter = params.spectral_norm_maxiter)
    norm_Q = quadratic_norm(Q, params)
    η0 = iszero(norm_A) ? one(T) : invnorm_scaling * inv(norm_A)
    # HPR-QP starts from σ = ‖b‖ / ‖c‖, which `primal_weight_init` inverts
    ω = batched_expand(sol.x, η0 .* primal_weight_init(milp, params))
    η = broadcast!!((_, η) -> η, batched_similar(ω), ω, η0)
    step_sizes = StepSizes(; η, ω, norm_A, norm_Q)
    aty = mul!(similar(sol.x), At, sol.y)
    λ_Q = norm_Q / invnorm_scaling^2
    scratch = Scratch(sol)
    err = KKTErrors(sol)
    (; relative_kkt) = algo.termination
    (; termination_norm) = algo.generic
    kkt_errors!(err, scratch, sol, milp, relative_kkt, termination_norm)  # TODO: count this KKT pass
    tol = algo.termination.infeasibility_tol
    stats = ConvergenceStats(err; ray = ray_buffer(sol, tol), starting_time)
    λ_Qv = proximal_diagonal(Q, convert(T, λ_Q), sol.x)
    return HPRQPState(;
        sol, z = copy(sol), z_anchor = copy(sol),
        w = copy(sol.x), w_bar = copy(sol.x), w_anchor = copy(sol.x),
        aty, aty_bar = copy(aty), aty_anchor = copy(aty),
        buf1 = similar(sol.x), buf2 = similar(sol.x), buf3 = similar(sol.x), buf4 = similar(sol.x),
        # with `η = invnorm_scaling / ‖A‖`, `σ λ_A = 1 / (η ω)`
        λ_A = inv(η0)^2, λ_Q = λ_Q,
        λ_Qv, Q_diagonal = is_diagonal_sparse(host_sparse(Q)),
        free_splitting = mostly_free(milp),
        golden = ntuple(_ -> batched_expand(sol.x, zero(T)), 10),
        fpe_reference = batched_expand(sol.x, convert(T, Inf)),
        t1 = batched_expand(sol.x, zero(T)), t2 = batched_expand(sol.x, zero(T)),
        t3 = batched_expand(sol.x, zero(T)),
        step_sizes, scratch, halpern = HalpernStats(sol, ω), stats,
        infeas = InfeasibilityWorkspace(sol, tol),
    )
end

"""
    solve!(state::HPRQPState, milp, algo)

Run the restarted Halpern Peaceman-Rachford method until termination: the fixed-point residual
is measured on the last step before each check, after which the instances which need it restart.
"""
function solve!(state::HPRQPState, milp::MILP, algo::Algorithm{:HPRQP})
    check_every = algo.generic.check_every
    check_every >= 2 || throw(ArgumentError("HPRQP needs `check_every >= 2`, got $check_every"))
    prog = init_progress("HPRQP iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        must_terminate = check_pass!(state, milp, algo, prog)
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    check_pass!(state::HPRQPState, milp, algo, prog = NoProgress())

Run one block of `check_every` steps (the first and the last one measuring the fixed-point residual), the
termination check and the restarts, and return whether to stop (see [`solve!`](@ref)).
"""
function check_pass!(
        state::HPRQPState,
        milp::MILP,
        algo::Algorithm{:HPRQP},
        prog = NoProgress()
    )
    # as in HPR-QP (`main_update!`), the reference residual of a restart cycle is that of its
    # first step, not the one which triggered the restart: hence a measured first step
    step!(state, milp, algo, Val(true))
    seed_fixed_point_error!(state.halpern)
    next_progress!(prog, state)
    @trace for _ in 1:(algo.generic.check_every - 2)
        step!(state, milp, algo, Val(false))
        next_progress!(prog, state)
    end
    step!(state, milp, algo, Val(true))
    next_progress!(prog, state)
    must_terminate = termination_check!(state, milp, algo)
    hprqp_restart_check!(state, algo, must_terminate)
    restart!(state, milp, algo)
    return must_terminate
end

"""
    step!(state::HPRQPState, milp, algo, Val(fixed_point_error))

Apply one step of HPR-QP (`kernels.jl` of HPR-QP), with `σ = η / ω`, `σ_d = η ω = 1 / (σ λ_A)`:

    x̄ = proj_[lv, uv](x + σ (Aᵀy - c - Q w))
    x̂ = 2x̄ - x
    w̄½ = (σ Λ w + x̂) / (1 + σ Λ)
    x̃ = x̂ + σ (Q w - Q w̄½)
    ȳ = y - σ_d A x̃ - σ_d proj_[-uc, -lc](y / σ_d - A x̃)
    w̄ = w̄½ + σ (Aᵀȳ - Aᵀy) / (1 + σ Λ)    (w̄ = w̄½ when more than 80% of the variables are free)

where `Λ` is `λ_Q I`, or `Diagonal(Q)` for a diagonal `Q`, and `x̃` is the symmetric Gauss-Seidel
correction of HPR-QP,

followed by the reflected Halpern combination of `(x, y, w, Aᵀy)` with their images and the
anchor. A step costs one product by each of `A` and `Aᵀ`, and two by `Q`. With `Val(true)`, the fixed-point
residual is also stored in `state.halpern.fpe`.
"""
function step!(
        state::HPRQPState{T}, milp::MILP{T}, algo::Algorithm{:HPRQP}, ::Val{fixed_point_error},
    ) where {T, fixed_point_error}
    (; sol, z, z_anchor, w, w_bar, w_anchor, aty, aty_bar, aty_anchor, λ_Qv) = state
    (; step_sizes, scratch, halpern) = state
    (; c, lv, uv, A, At, Q, lc, uc) = milp
    (; η, ω) = step_sizes
    σ = transpose(primal_dual_steps!!(/, scratch.b1, η, ω))
    σd = transpose(primal_dual_steps!!(*, scratch.b2, η, ω))

    (; buf1, buf2) = state
    # a frozen instance of a batch keeps `(x̄, ȳ)`, `z`, `w` and `Aᵀy` (see `active_mask`)
    active = active_mask(state.stats)
    Qw = mul_quadratic!(scratch.z, Q, w)
    masked_broadcast!(sol.x, active, z.x, σ, aty, c, Qw, lv, uv) do x, σ, aty, c, Qw, l, u
        clamp(x + σ * (aty - c - Qw), l, u)
    end
    x_hat = @. scratch.x = 2 * sol.x - z.x
    # first half of the `w` step; `σ` is per column and `λ_Qv` per row (under Reactant,
    # arithmetic between two traced scalars inside the elementwise kernel failed to compile)
    @. w_bar = (σ * λ_Qv * w + x_hat) / (1 + σ * λ_Qv)
    # symmetric Gauss-Seidel: the `y` step sees `x̂ + σ (Q w - Q w̄½)` rather than `x̂`, as in
    # `update_y_cpu!` of HPR-QP; without it, the iteration is not the one whose convergence
    # HPR-QP proves, and it oscillates (e.g. on DUAL2)
    Qw_half = mul_quadratic!(buf1, Q, w_bar)
    x_sgs = @. buf2 = x_hat + σ * (Qw - Qw_half)
    A_x_sgs = mul!(scratch.y, A, x_sgs)
    masked_dual_update!(sol.y, active, z.y, σd, A_x_sgs, lc, uc)
    mul!(aty_bar, At, sol.y)
    # second half of the `w` step, which HPR-QP's "noC" splitting for mostly free variables
    # leaves out (`unified_update_zxw_cpu!`)
    if !state.free_splitting
        @. w_bar += σ * (aty_bar - aty) / (1 + σ * λ_Qv)
    end

    if fixed_point_error
        halpern.fpe = hprqp_fixed_point_error!!(halpern.fpe, state, milp)
    end
    γ = algo.step_size.reflection_coefficient
    k = batched_apply!!(k -> (k + 1) / (k + 2), scratch.b1, halpern.inner)
    halpern_combine!(z.x, k, γ, sol.x, z_anchor.x, active)
    halpern_combine!(z.y, k, γ, sol.y, z_anchor.y, active)
    halpern_combine!(w, k, γ, w_bar, w_anchor, active)
    halpern_combine!(aty, k, γ, aty_bar, aty_anchor, active)
    halpern.inner = masked_apply!!(k -> k + 1, halpern.inner, active, halpern.inner)
    halpern.total += 1
    state.stats.kkt_passes += 1
    return nothing
end


"""
    hprqp_fixed_point_error!!(dest, state, milp)

Compute the fixed-point residual of the last step in the metric `M` of HPR-QP
(`compute_M_norm!` in `algorithm.jl` of HPR-QP), column by column, into `dest`. With
`Δ = T(z) - z`, `σ` the primal step and `Λ` the proximal diagonal (`λ_Q I`, or `Diagonal(Q)`):

    M₂ = ‖Δx‖² / σ - 2 ⟨Q Δw, Δx⟩ + 2 ⟨Aᵀ Δy, Δx⟩
    M₁ = σ λ_A ‖Δy‖² - 2σ ⟨Q Δw, Aᵀ Δy⟩ + σ ⟨Λ Δw, Q Δw⟩
    M₃ = σ² ⟨Aᵀ Δy, (I + σ Λ)⁻¹ Q Aᵀ Δy⟩
    ‖Δ‖_M = √(max(M₂ + max(M₁, 0), 0) + max(M₃, 0))

`Aᵀ Δy` comes from the carried `Aᵀy`, so this costs two products by `Q`.
"""
function hprqp_fixed_point_error!!(dest::BatchedNumber, state::HPRQPState, milp::MILP)
    (; sol, z, w, w_bar, aty, aty_bar) = state
    return hprqp_metric_norm!!(dest, state, milp.Q, sol.x, z.x, sol.y, z.y, w_bar, w, aty_bar, aty)
end

"""
    hprqp_metric_norm!!(dest, state, Q, x1, x0, y1, y0, w1, w0, aty1, aty0)

Compute `‖(x1, y1, w1) - (x0, y0, w0)‖_M` (see [`hprqp_fixed_point_error!!`](@ref)) into `dest`,
and store in `state.t1`, `state.t2` and `state.t3` the coefficients `a`, `b` and `c` of the model
of [`hprqp_sigma_update!`](@ref) for these differences.
"""
function hprqp_metric_norm!!(
        dest::BatchedNumber, state::HPRQPState, Q::AbstractMatrix,
        x1::AbstractVecOrMat, x0::AbstractVecOrMat, y1::AbstractVecOrMat, y0::AbstractVecOrMat,
        w1::AbstractVecOrMat, w0::AbstractVecOrMat, aty1::AbstractVecOrMat, aty0::AbstractVecOrMat,
    )
    (; buf1, buf2, buf3, buf4, λ_A, λ_Qv, step_sizes, scratch) = state
    (; η, ω) = step_sizes
    Δx = @. scratch.x = x1 - x0
    Δy = @. scratch.y = y1 - y0
    Δw = @. buf1 = w1 - w0
    AtΔy = @. buf3 = aty1 - aty0
    QΔw = mul_quadratic!(scratch.z, Q, Δw)

    # b = ‖Δx‖² and a = λ_A ‖Δy‖² - 2⟨QΔw, AᵀΔy⟩ + ⟨ΛΔw, QΔw⟩
    state.t2 = colsum!!(state.t2, @. buf2 = Δx^2)
    ΛΔwQΔw = colsum!!(scratch.b2, @. buf4 = λ_Qv * Δw * QΔw)
    Δy2 = colsum!!(scratch.b1, @. scratch.y = Δy^2)
    state.t1 = batched_apply!!((y2, l) -> λ_A * y2 + l, state.t1, Δy2, ΛΔwQΔw)
    QΔwAtΔy = colsum!!(scratch.b1, @. buf2 = QΔw * AtΔy)
    state.t1 = batched_apply!!((a, q) -> a - 2q, state.t1, state.t1, QΔwAtΔy)
    # M₂ = b / σ - 2⟨QΔw, Δx⟩ + 2⟨AᵀΔy, Δx⟩, while `scratch.x` still holds Δx
    dest = batched_apply!!((b, η, ω) -> b * ω / η, dest, state.t2, η, ω)
    dest = batched_apply!!((m, d) -> m - 2d, dest, dest, colsum!!(scratch.b1, @. buf2 = QΔw * Δx))
    dest = batched_apply!!((m, d) -> m + 2d, dest, dest, colsum!!(scratch.b1, @. buf2 = AtΔy * Δx))
    # M₁ = σ a, then c = ⟨AᵀΔy, QAᵀΔy⟩ and M₃ = σ² ⟨AᵀΔy, (I + σΛ)⁻¹ QAᵀΔy⟩
    dest = batched_apply!!(
        (m, a, η, ω) -> max(m + max(η / ω * a, zero(a)), zero(m)), dest, dest, state.t1, η, ω
    )
    QAtΔy = mul_quadratic!(scratch.z, Q, AtΔy)
    state.t3 = colsum!!(state.t3, @. buf2 = AtΔy * QAtΔy)
    σ = transpose(primal_dual_steps!!(/, scratch.b2, η, ω))
    m3 = colsum!!(scratch.b1, @. buf2 = AtΔy * QAtΔy / (1 + σ * λ_Qv))
    return batched_apply!!(dest, dest, m3, η, ω) do m, m3, η, ω
        sqrt(m + max((η / ω)^2 * m3, zero(m)))
    end
end

"""
    mostly_free(milp)

Whether more than 80% of the variable bounds (over all instances of a batch) are free, the
test of HPR-QP for its "noC" splitting.
"""
function mostly_free(milp::MILP)
    lv, uv = Array(milp.lv), Array(milp.uv)
    nfree = count(i -> isinf(lv[i]) & isinf(uv[i]), eachindex(lv, uv))
    return nfree > 0.8 * length(lv)
end

"""
    proximal_diagonal(Q, λ_Q, x)

Return the diagonal of the proximal operator `Λ ⪰ Q` of the `w` step of HPR-QP, as a vector like
the columns of `x`: `diag(Q)` if `Q` is a nonzero sparse matrix with entries on its diagonal only,
which makes the step exact, and `λ_Q` everywhere otherwise.

The test runs once, on a host copy of the sparsity structure (see [`host_sparse`](@ref)), so that
the iterates are the same with every matrix format and backend.
"""
function proximal_diagonal(Q::AbstractMatrix, λ_Q::Number, x::AbstractVecOrMat{T}) where {T}
    # in the precision of the iterates
    v = fill!(similar(x, T, size(x, 1)), λ_Q)
    Q_host = host_sparse(Q)
    if is_diagonal_sparse(Q_host)
        copyto!(v, Vector{T}(diag(Q_host)))
    end
    return v
end

"""
    host_sparse(Q)

Return a host `SparseMatrixCSC` copy of a sparse `Q` (of any format and backend), or `Q` itself
for any other matrix, for which [`is_diagonal_sparse`](@ref) is then `false`.
"""
host_sparse(Q::AbstractMatrix) = Q
host_sparse(Q::GPUSparseMatrix) = SparseMatrixCSC(Q)

"""
    is_diagonal_sparse(Q)

Return whether `Q` is a nonzero `SparseMatrixCSC` with entries on its diagonal only.
"""
is_diagonal_sparse(::AbstractMatrix) = false
function is_diagonal_sparse(Q::SparseMatrixCSC)
    isquadratic(Q) || return false
    return all(j -> all(==(j), view(rowvals(Q), nzrange(Q, j))), axes(Q, 2))
end

"""
    hprqp_restart_check!(state, algo, must_terminate)

Decide, instance by instance, whether to restart, with the rules of HPR-QP (`check_restart` in
`algorithm.jl`), on the fixed-point residual `r` of the last step:

- the first check always restarts, and records `r` as the reference of the long inner loops;
- sufficient decay: `r ≤ sufficient_decay * r₀`, where `r₀` is the residual at the last restart;
- necessary decay: `r ≤ necessary_decay * r₀` and `r` larger than at the previous check;
- long inner loop: more than half the iterations since the start (a fifth once `r` fell below a
  tenth of the reference) without restart.

The best residual so far and the primal weight it was reached with are recorded for
[`hprqp_sigma_update!`](@ref).
"""
function hprqp_restart_check!(state::HPRQPState, algo::Algorithm{:HPRQP}, must_terminate)
    (; halpern, step_sizes) = state
    (; sufficient_decay, necessary_decay) = algo.restart
    (; total) = halpern
    halpern.restart = broadcast!!(
        halpern.restart, halpern.fpe, halpern.fpe_initial, halpern.fpe_last, halpern.inner,
        state.fpe_reference,
    ) do r, r0, r_last, inner, ref
        first = isinf(ref)
        sufficient = r <= sufficient_decay * r0
        necessary = (r <= necessary_decay * r0) & (r > r_last)
        # integer and `T` arithmetic only: a `Float64` literal would promote a traced `Float32`
        # to `Float64`, which some Reactant backends cannot run
        long_inner_loop = ifelse(10 * r > ref, 2 * inner >= total, 5 * inner >= total)
        !must_terminate & (first | sufficient | necessary | long_inner_loop)
    end
    # a frozen instance of a batch never restarts
    halpern.restart = mask_and!!(halpern.restart, active_mask(state.stats))
    # the first check sets the reference, and every check updates the best residual
    halpern.best_ω = broadcast!!(
        halpern.best_ω, halpern.fpe, halpern.best_gap, state.fpe_reference, step_sizes.ω,
        halpern.best_ω,
    ) do r, best, ref, ω, best_ω
        ifelse(isinf(ref) | (r < best), ω, best_ω)
    end
    halpern.best_gap = broadcast!!(
        halpern.best_gap, halpern.fpe, halpern.best_gap, state.fpe_reference,
    ) do r, best, ref
        ifelse(isinf(ref) | (r < best), r, best)
    end
    state.fpe_reference = broadcast!!(state.fpe_reference, state.fpe_reference, halpern.fpe) do ref, r
        ifelse(isinf(ref), r, ref)
    end
    halpern.fpe_last = broadcast!!(halpern.fpe_last, halpern.restart, halpern.fpe) do restart, r
        ifelse(restart, oftype(r, Inf), r)
    end
    return halpern.restart
end

"""
    golden_sigma(a, b, c, d)

Minimize `f(σ) = a σ + b / σ + c σ² / (1 + d σ)` over `σ ∈ [1e-12, 1e12]` by golden-section
search on `log σ`, as HPR-QP does (on `σ` itself) to choose its penalty parameter. The number of
iterations is fixed, so that the search is a plain computation under Reactant.
"""
function golden_sigma(a::T, b::T, c::T, d::T) where {T <: Number}
    f(t) = (σ = exp(t); a * σ + b / σ + c * σ^2 / (1 + d * σ))
    φ = convert(T, (sqrt(5) - 1) / 2)
    lo, hi = convert(T, log(1.0e-12)), convert(T, log(1.0e12))
    t1, t2 = hi - φ * (hi - lo), lo + φ * (hi - lo)
    f1, f2 = f(t1), f(t2)
    for _ in 1:60
        right = f2 < f1
        lo, hi = ifelse(right, t1, lo), ifelse(right, hi, t2)
        t1n = ifelse(right, t2, hi - φ * (hi - lo))
        t2n = ifelse(right, lo + φ * (hi - lo), t1)
        f1n = ifelse(right, f2, f(t1n))
        f2n = ifelse(right, f(t2n), f1)
        t1, t2, f1, f2 = t1n, t2n, f1n, f2n
    end
    return exp((lo + hi) / 2)
end

"""
    golden_sigma_diagonal!!(state)

Minimize HPR-QP's per-coordinate model of the convergence rate for a diagonal `Q`
(`golden_Q_diag` in HPR-QP), column by column,

    f(σ) = a σ + b / σ + σ² Σᵢ (AᵀΔy)ᵢ (QAᵀΔy)ᵢ / (1 + σ Qᵢᵢ),

over `σ ∈ [1e-12, 1e12]`, by the golden-section search of [`golden_sigma`](@ref) on `log σ`,
with one reduction over the variables per iteration. It reads `a` and `b` from `state.t1` and
`state.t2`, and `AᵀΔy` and `QAᵀΔy` from `state.buf3` and `state.scratch.z`, as left by
[`hprqp_metric_norm!!`](@ref), and returns the minimizer per column.
"""
function golden_sigma_diagonal!!(state::HPRQPState{T}) where {T}
    (; t1, t2, buf2, buf3, buf4, λ_Qv, scratch) = state
    lo, hi, x1, x2, f1, f2, xn, fn, r, tmp = state.golden
    @. buf4 = buf3 * scratch.z  # (AᵀΔy)ᵢ (QAᵀΔy)ᵢ
    function f!!(dest, t)
        dest = batched_apply!!(exp, dest, t)
        σ = transpose(dest)
        s = colsum!!(scratch.b1, @. buf2 = buf4 / (1 + σ * λ_Qv))
        return batched_apply!!(dest, dest, t1, t2, s) do σ, a, b, s
            ϵ = convert(T, 1.0e-12)  # the same floors as `hprqp_sigma_update!`
            max(a, ϵ) * σ + max(b, ϵ) / σ + σ^2 * s
        end
    end
    φ = convert(T, (sqrt(5) - 1) / 2)
    lo = batched_apply!!(_ -> convert(T, log(1.0e-12)), lo, lo)
    hi = batched_apply!!(_ -> convert(T, log(1.0e12)), hi, hi)
    x1 = batched_apply!!((lo, hi) -> hi - φ * (hi - lo), x1, lo, hi)
    x2 = batched_apply!!((lo, hi) -> lo + φ * (hi - lo), x2, lo, hi)
    f1, f2 = f!!(f1, x1), f!!(f2, x2)
    for _ in 1:60
        # as in `golden_sigma`: move the bracket, then evaluate the one new point per column
        r = batched_apply!!((f1, f2) -> ifelse(f2 < f1, one(T), zero(T)), r, f1, f2)
        lo = batched_apply!!((r, x1, lo) -> ifelse(r > 0, x1, lo), lo, r, x1, lo)
        hi = batched_apply!!((r, x2, hi) -> ifelse(r > 0, hi, x2), hi, r, x2, hi)
        xn = batched_apply!!(xn, r, lo, hi) do r, lo, hi
            ifelse(r > 0, lo + φ * (hi - lo), hi - φ * (hi - lo))
        end
        fn = f!!(fn, xn)
        # going right: (x1, x2) ← (x2, new), going left: (x1, x2) ← (new, x1)
        tmp = batched_apply!!((r, x2, xn) -> ifelse(r > 0, x2, xn), tmp, r, x2, xn)
        x2 = batched_apply!!((r, x1, xn) -> ifelse(r > 0, xn, x1), x2, r, x1, xn)
        x1 = batched_apply!!(identity, x1, tmp)
        tmp = batched_apply!!((r, f2, fn) -> ifelse(r > 0, f2, fn), tmp, r, f2, fn)
        f2 = batched_apply!!((r, f1, fn) -> ifelse(r > 0, fn, f1), f2, r, f1, fn)
        f1 = batched_apply!!(identity, f1, tmp)
    end
    return batched_apply!!((lo, hi) -> exp((lo + hi) / 2), xn, lo, hi)
end

"""
    hprqp_sigma_update!(state, Q, constrained, err)

Update the primal step `σ` of the restarting instances as HPR-QP does (`update_sigma!` in
`algorithm.jl`), from the differences `Δ` between `T(z)` and the anchor (`constrained` says
whether the problem has constraint rows):

- `σ̂` minimizes `a σ + b / σ + c σ² / (1 + λ_Q σ)` (see [`golden_sigma`](@ref)), with
  `a = λ_A ‖Δy‖² - 2⟨QΔw, AᵀΔy⟩ + ⟨ΛΔw, QΔw⟩`, `b = ‖Δx‖²` and `c = ⟨AᵀΔy, QAᵀΔy⟩`, or is
  `√(b / a)` without constraints; for a diagonal `Q` with constraints, it minimizes HPR-QP's
  per-coordinate model instead, see [`golden_sigma_diagonal!!`](@ref);
- it is blended in log space with the best `σ` so far, with weight `exp(-0.05 r / r_best)`;
- near convergence, it is multiplied by a clamped power of the ratio of the dual and primal
  relative residuals.
"""
function hprqp_sigma_update!(
        state::HPRQPState{T}, Q::AbstractMatrix, constrained::Bool, err::KKTErrors,
    ) where {T}
    (; sol, z_anchor, w_bar, w_anchor, aty_bar, aty_anchor, λ_Q, halpern, step_sizes) = state
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    # fills `t1`, `t2` and `t3` with `a`, `b` and `c`; the norm itself is not needed
    hprqp_metric_norm!!(
        state.scratch.b1, state, Q, sol.x, z_anchor.x, sol.y, z_anchor.y,
        w_bar, w_anchor, aty_bar, aty_anchor,
    )
    d = convert(T, λ_Q)
    diagonal = state.Q_diagonal
    # with a diagonal `Q`, HPR-QP minimizes the per-coordinate model instead (`golden_Q_diag`)
    σ_diag = if diagonal && constrained
        golden_sigma_diagonal!!(state)
    else
        state.golden[1]
    end
    step_sizes.ω = broadcast!!(
        step_sizes.ω, halpern.restart, state.t1, state.t2, state.t3, σ_diag, step_sizes.η,
        step_sizes.ω, halpern.fpe, halpern.best_gap, halpern.best_ω, primal, primal_scale, dual,
        dual_scale, gap, gap_scale,
    ) do restart, a, b, c, σ_diag, η, ω, r, best, best_ω, p, ps, rd_, ds, g, gs
        a, b = max(a, convert(T, 1.0e-12)), max(b, convert(T, 1.0e-12))
        σ_est = ifelse(constrained & !diagonal, golden_sigma(a, b, c, d), sqrt(b / a))
        σ_est = ifelse(constrained & diagonal, σ_diag, σ_est)
        fact = exp(-convert(T, 0.05) * r / best)
        σ_cand = exp(fact * log(σ_est) + (1 - fact) * log(η / best_ω))
        rp, rd, rg = p / ps, rd_ / ds, g / gs
        level = max(min(rd, rp), min(rg, r))
        ratio = rd / rp
        lo_κ, hi_κ = convert(T, 1.0e-2), convert(T, 100)
        κ = ifelse(
            level > convert(T, 9.0e-10), one(T),
            ifelse(
                level > convert(T, 5.0e-10), clamp(sqrt(ratio), lo_κ, hi_κ), clamp(ratio, lo_κ, hi_κ)
            ),
        )
        σ_new = κ * σ_cand
        ifelse(restart & isfinite(σ_new) & (σ_new > 0), η / σ_new, ω)
    end
    return nothing
end

"""
    restart!(state::HPRQPState, milp, algo)

Restart the instances selected by [`hprqp_restart_check!`](@ref): update their primal step with
[`hprqp_sigma_update!`](@ref), then move the iterate and the anchor to the image of the last
step, `w` and `Aᵀy` included, and take the current residual as the reference of the next cycle.
"""
function restart!(state::HPRQPState, milp::MILP, algo::Algorithm{:HPRQP})
    (; sol, z, z_anchor, w, w_bar, w_anchor, aty, aty_bar, aty_anchor) = state
    (; halpern, stats) = state
    (; restart) = halpern
    # the pieces of `milp` rather than `milp` itself: DispatchDoctor's stability check of the
    # deeper call chain did not fold with the whole problem in the signature, and then it ran
    # (and boxed `milp`) at every restart
    # the σ model costs two products by `Q`: skip it when nobody restarts, except inside a
    # compilation, where `restart` is only known at run time (the update is a no-op then)
    if within_compile() || !batched_all(!, restart)
        hprqp_sigma_update!(state, milp.Q, nbcons(milp) > 0, stats.err)
    end
    batched_select!(z_anchor, restart, sol)
    batched_select!(z, restart, sol)
    r = transpose(restart)
    broadcast!!(ifelse, w_anchor, r, w_bar, w_anchor)
    broadcast!!(ifelse, w, r, w_bar, w)
    broadcast!!(ifelse, aty_anchor, r, aty_bar, aty_anchor)
    broadcast!!(ifelse, aty, r, aty_bar, aty)
    halpern.inner = broadcast!!(halpern.inner, restart, halpern.inner) do r, k
        ifelse(r, zero(k), k)
    end
    halpern.restarts = broadcast!!(halpern.restarts, restart, halpern.restarts) do r, n
        ifelse(r, n + 1, n)
    end
    # the reference residual is refreshed at the first step of the next cycle
    halpern.fpe_pending = broadcast!!(identity, halpern.fpe_pending, restart)
    return nothing
end
