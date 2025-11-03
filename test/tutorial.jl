# # Tutorial

using CoolPDLP
using JuMP
using HiGHS
#md using UnicodePlots
using Test  #src

# ## Reading a MILP

# You can use [`read_milp`](@ref) to read a MILP in MPS format.
# Alternately, you can import a problem from a benchmark set (either MIPLIB 2017 or Netlib).

netlib = list_netlib_instances()
instance_name = first(netlib)

# The result is a [`MILP`](@ref) object.

milp, path = read_netlib_instance(instance_name)
milp

# ## Solving a MILP

# You can use the primal-dual hybrid gradient algortithm to solve a MILP.
# The first thing to do is define parameters.

params = PDHGParameters(;
    tol_termination = 1.0e-4,
    time_limit = 10.0,
    record_error_history = true
)

# Then all it takes is to call [`pdhg`](@ref).

final_state = pdhg(milp, params; show_progress = false)

# You can then plot the KKT relative error to show that it has decreased.

#md lineplot(
#md     first.(final_state.relative_error_history),
#md     last.(final_state.relative_error_history);
#md     xlabel = "KKT passes",
#md     ylabel = "Relative KKT error",
#md     title = "Convergence of PDHG on $instance_name instance"
#md )

# Tests, excluded from Markdown output  #src

first_err = first(final_state.relative_error_history)[2]  #src
last_err = last(final_state.relative_error_history)[2]  #src
@test last_err < first_err  #src
