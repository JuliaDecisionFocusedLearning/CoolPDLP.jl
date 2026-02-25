# # Tutorial

using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using JuMP: JuMP, MOI
using MathOptBenchmarkInstances: Netlib, list_instances, read_instance
using Test  #src

# ## Creating a MILP

# You can use [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to read a MILP from a local MPS file, or [MathOptBenchmarkInstances.jl](https://github.com/JuliaDecisionFocusedLearning/MathOptBenchmarkInstances.jl) to automatically download standard benchmark sets (which we do here).

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

# Note that manual construction is also an option if you provide the constraints, variable bounds and objectives as arrays.

# ## Solving a MILP

# You can use the PDLP algortithm to solve the continuous relaxation of a MILP.
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

# ## Running on the GPU

# To run the same algorithm on the GPU, all it takes is to define a different set of parameters and thus force conversion of the instance:

algo_gpu = PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    GPUSparseMatrixCSR;  # GPU sparse matrix type, replace with e.g. CuSparseMatrixCSR
    backend = JLBackend(),  # replace with e.g. CUDABackend()
    termination_reltol = 1.0f-6,
    time_limit = 10.0,
)

# The result of the algorithm will live on the GPU:

sol_gpu, stats_gpu = solve(milp, algo_gpu)
sol_gpu.x

# To bring in back to the CPU, just call the `Array` converter.

objective_value(Array(sol_gpu.x), milp)

# ## Using the JuMP interface

# If you have a model available in [JuMP.jl](https://github.com/jump-dev/JuMP.jl), you can simply choose [`CoolPDLP.Optimizer`](@ref) as your solver, and pass it the same options as before.

model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
JuMP.set_optimizer(model, CoolPDLP.Optimizer)
JuMP.set_silent(model)
JuMP.set_attribute(model, "termination_reltol", 1.0e-6)
JuMP.set_attribute(model, "matrix_type", GPUSparseMatrixCSR)
JuMP.set_attribute(model, "backend", JLBackend())
JuMP.optimize!(model)
x_jump = JuMP.value.(JuMP.all_variables(model))

#-

objective_value(x_jump, milp)

# ## Comparing with HiGHS

# To check our solution, let's compare it with the output from the HiGHS solver:

model_highs = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
JuMP.set_optimizer(model_highs, HiGHS.Optimizer)
JuMP.set_silent(model_highs)
JuMP.optimize!(model_highs)
x_ref = JuMP.value.(JuMP.all_variables(model_highs))

# Of course, the solution given by HiGHS is feasible too, and we can compare objective values:

is_feasible(x_ref, milp; cons_tol = 1.0e-4)

#-

objective_value(x_ref, milp)

# Tests, excluded from Markdown output  #src

first_err = CoolPDLP.relative(first(stats.error_history)[2])  #src
last_err = CoolPDLP.relative(last(stats.error_history)[2])  #src
@test last_err < first_err  #src
@test is_feasible(sol.x, milp; cons_tol = 1.0e-3)  #src
@test is_feasible(Array(sol_gpu.x), milp; cons_tol = 1.0e-3)  #src
@test is_feasible(x_jump, milp; cons_tol = 1.0e-3)  #src
@test is_feasible(x_ref, milp)  #src
@test objective_value(sol.x, milp) ≈ objective_value(x_ref, milp) rtol = 1.0e-3  #src
@test objective_value(Array(sol_gpu.x), milp) ≈ objective_value(x_ref, milp) rtol = 1.0e-3  #src
@test objective_value(Array(x_jump), milp) ≈ objective_value(x_ref, milp) rtol = 1.0e-3  #src
