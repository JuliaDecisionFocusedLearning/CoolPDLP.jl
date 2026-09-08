using CoolPDLP
using CoolPDLP: custom_to_rarray
using MathOptBenchmarkInstances
using Reactant
using Reactant: to_rarray
using Test

dataset = Netlib
list = list_instances(dataset);
name = list[1]
qps, path = read_instance(dataset, name);

milp0 = MILP(qps; dataset, name, path)
sol0 = PrimalDualSolution(milp0);

@testset verbose = true "$(typeof(algo))" for algo in [
        PDHG(
            Float32,
            Int32,
            Matrix;
            backend = nothing,
            termination_reltol = 1.0f-3,
            time_limit = 1.0,
            record_error_history = false,
            show_progress = false
        ),
        PDLP(
            Float32,
            Int32,
            Matrix;
            backend = nothing,
            termination_reltol = 1.0f-3,
            time_limit = 1.0,
            record_error_history = false,
            show_progress = false
        ),
    ]
    milp, sol = preprocess(milp0, sol0, algo)
    state = initialize(milp, sol, algo; starting_time = time())

    milp_r = to_rarray(milp; track_numbers = true)
    state_r = to_rarray(state; track_numbers = true)
    algo_r = to_rarray(algo; track_numbers = true)

    @test_nowarn compiled_step! = @compile CoolPDLP.step!(state_r, milp_r)
    @test_nowarn compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)

    compiled_step! = @compile CoolPDLP.step!(state_r, milp_r)
    compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)

    @test_nowarn compiled_step!(deepcopy(state_r), milp_r)
    @test_nowarn compiled_solve!(deepcopy(state_r), milp_r, algo_r)
end
