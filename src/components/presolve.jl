function presolve(
        milp::MILP;
        input_instance::String = mktemp(),
        postsolve_file::String = mktemp(),
        presolved_instance::String = mktemp(),
    )
    (; c, lv, uv, lc, uc, A) = milp
    @assert !isbatched(milp)
    model = Model()
    @variable(model, x[1:nbvar(milp)])
    @constraint(model, lv .<= x .<= uv)
    @constraint(model, lc .<= A * x .<= uc)
    @objective(model, dot(c, x))
    dest = MOI.FileFormats.Model(
        format = MOI.FileFormats.FORMAT_MPS
    )
    MOI.copy_to(dest, model)
    MOI.write_to_file(dest, input_instance)
    return presolved_instance, postsolve_file
end

function postsolve(
        sol::PrimalDualSolution;
        postsolve_file::String,
        reduced_sol::String = mktemp(),
        original_sol::String = mktemp(),
    )

    # write `.sol` file `reduced_sol`

    return postsolve_from_file(postsolve_file, reduced_sol, original_sol)

    # read `.sol` from file `original_sol`

end
