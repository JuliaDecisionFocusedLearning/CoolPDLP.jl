using CoolPDLP
using CoolPDLP: KKTErrors
using MathOptBenchmarkInstances
using Reactant
using Reactant: to_rarray
using Test

# The same test suite runs on CPU and on GPU; the Buildkite `cuda` queue sets this to "gpu".
const REACTANT_BACKEND = get(ENV, "COOLPDLP_REACTANT_BACKEND", "cpu")
Reactant.set_default_backend(REACTANT_BACKEND)
const REACTANT_PLATFORM = Reactant.XLA.platform_name(Reactant.XLA.default_backend())
@info "Running Reactant tests" REACTANT_BACKEND REACTANT_PLATFORM Reactant.devices()

@testset "Requested backend is in use" begin
    # Reactant falls back to the CPU client when no accelerator is available, which would
    # let the GPU job pass green without ever touching the device
    if REACTANT_BACKEND == "cpu"
        @test lowercase(REACTANT_PLATFORM) == "cpu"
    else
        @test lowercase(REACTANT_PLATFORM) != "cpu"
    end
end

dataset = Netlib
list = list_instances(dataset);
name = list[1]
qps, path = read_instance(dataset, name);

milp0 = MILP(qps; dataset, name, path)
sol0 = PrimalDualSolution(milp0);

"""
    unwrap(x)

Bring a scalar or array back to plain Julia, whether or not it comes from a Reactant state.

Reactant numbers are widened to `Float64` so that a compiled `Float32` result can be
compared with its plain counterpart without going through `Reactant` conversions.
"""
unwrap(x::Number) = Float64(x)
unwrap(x::AbstractArray) = Array(x)

"""
    solve_plain_and_compiled(algo)

Solve the same problem twice with `algo`, from the same starting point: once with the plain
Julia loop, once with a Reactant-compiled one.

Return the two final states, whose contents should agree up to numerical precision.
"""
function solve_plain_and_compiled(algo::CoolPDLP.Algorithm)
    milp, sol = preprocess(milp0, sol0, algo)
    state = initialize(milp, sol, algo; starting_time = time())
    solve!(state, milp, algo)

    # `solve!` mutates both the state and the scratch space it shares with the problem, so
    # the compiled run starts from its own copy of everything
    milp_copy, sol_copy = preprocess(milp0, sol0, algo)
    state_copy = initialize(milp_copy, sol_copy, algo; starting_time = time())
    milp_r = to_rarray(milp_copy; track_numbers = true)
    state_r = to_rarray(state_copy; track_numbers = true)
    algo_r = to_rarray(algo; track_numbers = true)
    compiled_solve! = @compile solve!(state_r, milp_r, algo_r)
    compiled_solve!(state_r, milp_r, algo_r)

    return state, state_r
end

# `time_limit` is deliberately left out. The compiled loop does read the clock at each check
# now, so a binding time limit would stop the two runs after different numbers of iterations
# and make them incomparable. The KKT pass budget bounds the runtime instead.
configs = [
    (:PDHG, Float32, 1.0f-2, 1000, 1.0e-3),
    (:PDLP, Float32, 1.0f-2, 1000, 1.0e-3),
    (:PDHG, Float64, 1.0e-4, 2000, 1.0e-8),
    (:PDLP, Float64, 1.0e-4, 2000, 1.0e-8),
]

@testset verbose = true "$A in $T" for (A, T, termination_reltol, max_kkt_passes, rtol) in configs
    algo = CoolPDLP.Algorithm{A}(
        T,
        Int32,
        Matrix;
        backend = nothing,
        termination_reltol,
        max_kkt_passes,
        time_limit = Inf,
        check_every = 50,
        record_error_history = false,
        show_progress = false,
    )
    state, state_r = solve_plain_and_compiled(algo)

    @testset "Finite iterates" begin
        @test all(isfinite, unwrap(state_r.sol.x))
        @test all(isfinite, unwrap(state_r.sol.y))
    end

    @testset "Same number of KKT passes" begin
        # both loops must stop at the same point for the iterates to be comparable at all
        @test unwrap(state_r.stats.kkt_passes) == unwrap(state.stats.kkt_passes)
    end

    @testset "Same solution" begin
        @test isapprox(unwrap(state_r.sol.x), unwrap(state.sol.x); rtol)
        @test isapprox(unwrap(state_r.sol.y), unwrap(state.sol.y); rtol)
    end

    @testset "Same KKT errors" begin
        @testset "$field" for field in fieldnames(KKTErrors)
            err = unwrap(getfield(state.stats.err, field))
            err_r = unwrap(getfield(state_r.stats.err, field))
            @test isapprox(err_r, err; rtol)
        end
    end
end

# Reading the host clock inside a compiled program needs a Reactant callback, which not every
# backend can service; the extension falls back to the frozen trace-time clock when it cannot.
const REACTANT_EXT = Base.get_extension(CoolPDLP, :CoolPDLPReactantExt)

if !REACTANT_EXT.host_callbacks_supported()
    @info "Skipping the time limit tests: this backend cannot run a host callback" REACTANT_PLATFORM
else
    @testset verbose = true "Time limit" begin
        # `termination_reltol` is unreachable and the KKT budget is far away, so the time limit is
        # the only thing that can stop this solve. The budget is still finite, to bound the damage
        # if the limit stops working.
        time_limit = 0.5
        max_kkt_passes = 20_000
        algo = PDLP(
            Float64,
            Int64,
            Matrix;
            backend = nothing,
            termination_reltol = 1.0e-12,
            time_limit,
            max_kkt_passes,
            check_every = 50,
            record_error_history = false,
            show_progress = false,
        )
        milp, sol = preprocess(milp0, sol0, algo)
        state = initialize(milp, sol, algo; starting_time = time())

        milp_r = to_rarray(milp; track_numbers = true)
        state_r = to_rarray(state; track_numbers = true)
        algo_r = to_rarray(algo; track_numbers = true)

        compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)
        # compilation happens after `initialize`, and counts against the time limit like any other
        # elapsed time, so restart the clock now that it is over
        state_r.stats.starting_time = to_rarray(time(); track_numbers = true)
        compiled_solve!(state_r, milp_r, algo_r)

        elapsed = unwrap(state_r.stats.time_elapsed)
        passes = unwrap(state_r.stats.kkt_passes)

        @testset "The clock advances inside the compiled loop" begin
            # without the host callback, `time()` is folded to its trace-time value and the elapsed
            # time is a constant fixed before the run, here a negative one
            @test elapsed > 0
        end

        @testset "The solve stops on the time limit" begin
            @test elapsed >= time_limit
            @test passes < max_kkt_passes
        end
    end
end
