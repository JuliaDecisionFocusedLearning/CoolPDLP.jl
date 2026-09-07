using CoolPDLP
using CoolPDLP: custom_to_rarray
using MathOptBenchmarkInstances
using Reactant
using Test

dataset = Netlib
list = list_instances(dataset);
name = list[end]
qps, path = read_instance(dataset, name);

milp0 = MILP(qps; dataset, name, path)
sol0 = PrimalDualSolution(milp0);

algo = PDHG(
    Float32,
    Int32,
    Matrix;
    backend = nothing,
    termination_reltol = 1.0f-6,
    time_limit = 10.0,
    record_error_history = false,
    show_progress = false
);

milp, sol = preprocess(milp0, sol0, algo);
state = initialize(milp, sol, algo; starting_time = time());

milp_r = CoolPDLP.custom_to_rarray(milp);
state_r = CoolPDLP.custom_to_rarray(state; track_numbers = true);
algo_r = CoolPDLP.custom_to_rarray(algo; track_numbers = true)

conversion_r = custom_to_rarray(algo.conversion; track_numbers = true) |> typeof
preconditioning_r = custom_to_rarray(algo.preconditioning; track_numbers = true) |> typeof
step_size_r = custom_to_rarray(algo.step_size; track_numbers = true) |> typeof
restart_r = custom_to_rarray(algo.restart; track_numbers = true) |> typeof
generic_r = custom_to_rarray(algo.generic; track_numbers = true) |> typeof
termination_r = custom_to_rarray(algo.termination; track_numbers = true) |> typeof

compiled_step! = @compile CoolPDLP.step!(state_r, milp_r)
compiled_solve! = @compile CoolPDLP.solve!(state_r, milp_r, algo_r)

using Chairmarks
@be initialize(milp, sol, algo; starting_time = time()) CoolPDLP.step!(_, milp) seconds = 10
@be CoolPDLP.custom_to_rarray(state; track_numbers = true) compiled_step!(_, milp_r) seconds = 10
