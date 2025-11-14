abstract type AbstractProblem end

"""
    nbvar(problem)

Return the number of variables in `problem`.
"""
function nbvar end

"""
    nbvar_int(problem)

Return the number of integer variables in `problem`.
"""
function nbvar_int end

"""
    nbvar_cont(problem)

Return the number of integer variables in `problem`.
"""
function nbvar_cont end

"""
    nbcons(problem)

Return the number of constraints in `problem`, not including variable bounds or integrality requirements.
"""
function nbcons end

"""
    nbcons_eq(problem)

Return the number of equality constraints in `problem`.
"""
function nbcons_eq end

"""
    nbcons_ineq(problem)

Return the number of inequality constraints in `problem`, not including variable bounds.
"""
function nbcons_ineq end
