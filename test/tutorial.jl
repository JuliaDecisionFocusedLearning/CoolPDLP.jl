# # Tutorial

using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using JuMP: JuMP, MOI
using MathProgBenchmarks: Netlib, list_instances, read_instance
using Test  #src

# ## Reading a MILP

# You can use [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) to read a MILP from a local MPS file, or [MathProgBenchmarks.jl](https://github.com/JuliaDecisionFocusedLearning/MathProgBenchmarks.jl) to automatically download standard benchmark sets.

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

# You can use the PDHG algortithm to solve a MILP.
# The first thing to do is define parameters.

params = PDHGParameters(;
    termination_reltol = 1.0e-6,
    time_limit = 10.0,
)

# Then all it takes is to call [`pdhg`](@ref).

(x, y), stats = pdhg(milp, params; show_progress = false);

# The solution is available as a primal-dual pair `(x, y)`:

x

# The stats contain information about the convergence of the algorithm:

stats

# You can check the feasibility and objective value:

is_feasible(x, milp; cons_tol = 1.0e-4)

#-

objective_value(x, milp)

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

params_gpu = PDHGParameters(
    Float32,  # desired float type
    Int32,  # desired int type
    GPUSparseMatrixCSR,  # hardware-agnostic GPU sparse matrix
    JLBackend();  # replace with e.g. CUDABackend()
    termination_reltol = 1.0f-6,
    time_limit = 10.0,
)

# The result of the algorithm will live on the GPU:

(x_gpu, y_gpu), stats_gpu = pdhg(milp, params_gpu; show_progress = false)
x_gpu

# To bring in back to the CPU, just call the `Array` converter.

objective_value(Array(x_gpu), milp)

# Tests, excluded from Markdown output  #src

first_err = CoolPDLP.relative(first(stats.error_history)[2])  #src
last_err = CoolPDLP.relative(last(stats.error_history)[2])  #src
@test last_err < first_err  #src
@test is_feasible(x, milp; cons_tol = 1.0e-3)  #src
@test is_feasible(Array(x_gpu), milp; cons_tol = 1.0e-3)  #src
@test is_feasible(x_jump, milp)  #src
@test objective_value(x, milp) ≈ objective_value(x_jump, milp) rtol = 1.0e-3  #src
@test objective_value(Array(x_gpu), milp) ≈ objective_value(x_jump, milp) rtol = 1.0e-3  #src
