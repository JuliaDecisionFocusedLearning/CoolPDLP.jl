using CoolPDLP
using MathOptBenchmarkInstances
using Reactant
using Test

dataset = Netlib
list = list_instances(dataset);
name = list[4]
qps, path = read_instance(dataset, name);

milp0 = MILP(qps; dataset, name, path);
sol0 = PrimalDualSolution(milp0);

algo = PDHG(
    Float32,
    Int32,
    Matrix;
    termination_reltol = 1.0f-6,
    time_limit = 10.0,
    record_error_history = false,
    show_progress = false
);

milp, sol = preprocess(milp0, sol0, algo);
state = initialize(milp, sol, algo; starting_time = time());

milp_r = CoolPDLP.custom_to_rarray(milp);
state_r = CoolPDLP.custom_to_rarray(state; track_numbers = true);

compile_options = CompileOptions(; donated_args = :none)
@test_nowarn compiled_step! = @compile compile_options = compile_options CoolPDLP.step!(state_r, milp_r)
