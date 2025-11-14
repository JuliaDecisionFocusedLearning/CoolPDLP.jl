using CoolPDLP
using CoolPDLP: nbcons_ineq, nbcons_eq
using JuMP: JuMP, MOI
using HiGHS: HiGHS
using Test

function jump_nbcons(model)
    eq, ineq = 0, 0
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: JuMP.AffExpr || continue
        if S <: MOI.EqualTo
            eq += JuMP.num_constraints(model, F, S)
        elseif S <: MOI.GreaterThan || S <: MOI.LessThan
            ineq += JuMP.num_constraints(model, F, S)
        elseif S <: MOI.Interval
            ineq += 2 * JuMP.num_constraints(model, F, S)
        else
            error("constraint type not handled")
        end
    end
    return (; eq, ineq)
end

@testset "Netlib" begin
    netlib = list_netlib_instances()
    @testset for instance_name in netlib
        milp = read_netlib_instance(instance_name)
        if instance_name in ["agg", "blend", "dfl001", "forplan", "gfrd-pnc", "sierra"]
            @test_skip JuMP.read_from_file(milp.path; format = MOI.FileFormats.FORMAT_MPS)
        else
            jump_model = JuMP.read_from_file(milp.path; format = MOI.FileFormats.FORMAT_MPS)
            @test nbvar(milp) == JuMP.num_variables(jump_model)
            @test nbcons_eq(milp) == jump_nbcons(jump_model).eq
            @test nbcons_ineq(milp) == jump_nbcons(jump_model).ineq
        end
    end
end;
