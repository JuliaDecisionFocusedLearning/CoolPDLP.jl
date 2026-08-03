using Test
import MathOptInterface as MOI
import CoolPDLP
import JuMP

function test_moi(matrix_type, backend, ::Type{T} = Float64) where {T <: AbstractFloat}
    model = JuMP.Model(CoolPDLP.Optimizer{T})
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "matrix_type", matrix_type)
    JuMP.set_attribute(model, "backend", backend)
    JuMP.set_attribute(model, "time_limit", 1000.0)

    JuMP.@variable(model, x >= T(0))
    JuMP.@variable(model, T(0) <= y <= T(3))
    JuMP.@objective(model, Min, T(12) * x + T(20) * y)
    JuMP.@constraint(model, c1, T(6) * x + T(8) * y >= T(100))
    JuMP.@constraint(model, c2, T(7) * x + T(12) * y >= T(120))
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ 205.0 atol = 1.0e-2

    return nothing
end
