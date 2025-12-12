# # Tutorial

using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using JuMP: JuMP, MOI
using MathOptBenchmarkInstances: Netlib, list_instances, read_instance
using Test  #src

# ## Reading a MILP

# You can use [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to read a MILP from a local MPS file, or [MathOptBenchmarkInstances.jl](https://github.com/JuliaDecisionFocusedLearning/MathOptBenchmarkInstances.jl) to automatically download standard benchmark sets.

dataset = Netlib
list = list_instances(dataset)
name = list[4]
qps, path = read_instance(dataset, name);

# A [`MILP`](@ref) object can be constructed from there:

milp = MILP(qps; dataset, name, path)

# Its attributes can be queried:

nbvar(milp)

#-

nbcons(milp)

# ## Solving a MILP

# You can use the PDLP algortithm to solve a MILP.
# The first thing to do is define parameters inside a [`PDLP`](@ref) struct.

algo = PDLP(;
    termination_reltol = 1.0e-6,
    time_limit = 10.0,
)

# Then all it takes is to call [`solve`](@ref).

sol, stats = solve(milp, algo);

# The solution is available as a [`PrimalDualSolution`](@ref):

sol.x

# The stats contain information about the convergence of the algorithm:

stats

# You can check the feasibility and objective value:

is_feasible(sol.x, milp; cons_tol = 1.0e-4)

#-

objective_value(sol.x, milp)

# ## Comparing with JuMP

# Here is how you can compare your result with the solution to a JuMP model:

jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
JuMP.set_silent(jump_model)
JuMP.optimize!(jump_model)
x_jump = JuMP.value.(JuMP.all_variables(jump_model))

# Of course, that solution is feasible too, and we can compare objective values:

is_feasible(x_jump, milp; cons_tol = 1.0e-4)

#-

objective_value(x_jump, milp)

# ## Running on the GPU

# To run the same algorithm on the GPU, all it takes is to define a different set of parameters and thus force conversion of the instance:

algo_gpu = PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    GPUSparseMatrixCSR;  # hardware-agnostic GPU sparse matrix
    backend = JLBackend(),  # replace with e.g. CUDABackend()
    termination_reltol = 1.0f-6,
    time_limit = 10.0,
    show_progress = false,
)

# The result of the algorithm will live on the GPU:

sol_gpu, stats_gpu = solve(milp, algo_gpu)
sol_gpu.x

# To bring in back to the CPU, just call the `Array` converter.

objective_value(Array(sol_gpu.x), milp)

# Tests, excluded from Markdown output  #src

first_err = CoolPDLP.relative(first(stats.error_history)[2])  #src
last_err = CoolPDLP.relative(last(stats.error_history)[2])  #src
@test last_err < first_err  #src
@test is_feasible(sol.x, milp; cons_tol = 1.0e-3)  #src
@test is_feasible(Array(sol_gpu.x), milp; cons_tol = 1.0e-3)  #src
@test is_feasible(x_jump, milp)  #src
@test objective_value(sol.x, milp) ≈ objective_value(x_jump, milp) rtol = 1.0e-3  #src
@test objective_value(Array(sol_gpu.x), milp) ≈ objective_value(x_jump, milp) rtol = 1.0e-3  #src
