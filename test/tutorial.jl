# # Tutorial

using CoolPDLP
using JuMP: JuMP, MOI
using HiGHS: HiGHS
using Test  #src

# ## Reading a MILP

# You can use [`read_milp`](@ref) to read a MILP in MPS format.
# Alternately, you can import a problem from a benchmark set (either MIPLIB 2017 or Netlib).

netlib = list_netlib_instances()
instance_name = netlib[4]

# The result is a [`MILP`](@ref) object.

milp = read_netlib_instance(instance_name)

# ## Solving a MILP

# You can use the PDLP algortithm to solve a MILP.
# The first thing to do is define parameters.

params = PDLPParameters(;
    termination_reltol = 1.0e-6,
    time_limit = 10.0,
    record_error_history = true
)

# Then all it takes is to call [`pdlp`](@ref).

sol, final_state = pdlp(milp, params; show_progress = false);

# The solution is available as a primal-dual pair:

sol.x'

# The final state contains information about the convergence of the algorithm:

final_state.termination_reason

# You can check its feasibility and objective value:

is_feasible(sol.x, milp; cons_tol = 1.0e-3)

#-

objective_value(sol.x, milp)

# ## Comparing with JuMP

# Here is how you can compare your result with the solution to a JuMP model:

jump_model = JuMP.read_from_file(milp.path; format = MOI.FileFormats.FORMAT_MPS)
JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
JuMP.set_silent(jump_model)
JuMP.optimize!(jump_model)
x_jump = JuMP.value.(JuMP.all_variables(jump_model))

# Of course, that solution is feasible too, and we can compare objective values:

is_feasible(x_jump, milp; cons_tol = 1.0e-4)

#-

objective_value(x_jump, milp)

# Tests, excluded from Markdown output  #src

first_err = first(final_state.error_history)[2].rel_max  #src
last_err = last(final_state.error_history)[2].rel_max  #src
@test last_err < first_err  #src
@test is_feasible(sol.x, milp; cons_tol = 1.0e-3)  #src
@test is_feasible(x_jump, milp)  #src
@test objective_value(sol.x, milp) ≈ objective_value(x_jump, milp) rtol = 1.0e-3  #src
