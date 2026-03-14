using Test
import MathOptInterface as MOI
import CoolPDLP
using CUDA
using CUDA.CUSPARSE
import JuMP

model = JuMP.Model(CoolPDLP.Optimizer)
JuMP.set_silent(model)
JuMP.set_attribute(model, "matrix_type", CUSPARSE.CuSparseMatrixCSR)
JuMP.set_attribute(model, "backend", CUDABackend())

JuMP.@variable(model, x >= 0)
JuMP.@variable(model, 0 <= y <= 3)
JuMP.@objective(model, Min, 12x + 20y)
JuMP.@constraint(model, c1, 6x + 8y >= 100)
JuMP.@constraint(model, c2, 7x + 12y >= 120)
JuMP.optimize!(model)
@test JuMP.termination_status(model) == MOI.OPTIMAL
@test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
@test JuMP.objective_value(model) ≈ 205.0 atol = 1.0e-2
