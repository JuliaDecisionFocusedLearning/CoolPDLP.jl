using Test
import MathOptInterface as MOI
import CoolPDLP
using CUDA
using CUDA.CUSPARSE
import JuMP
using JLArrays: JLBackend

@testset "MOI Test Suite" begin
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            CoolPDLP.Optimizer(),
        ),
        Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            atol = 1.0e-3,
            rtol = 1.0e-3,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.ObjectiveBound,
                MOI.VariableBasisStatus,
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
            ],
        );
        exclude = [
            # TODO: infeasible/unbounded detection
            r"INFEASIB",
            r"unbounded",
            # Poor precision on this test
            r"test_linear_add_constraints",
        ],
    )
end

@testset "JLBackend" begin
    model = JuMP.Model(CoolPDLP.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "matrix_type", CoolPDLP.GPUSparseMatrixCSR)
    JuMP.set_attribute(model, "backend", JLBackend())

    JuMP.@variable(model, x >= 0)
    JuMP.@variable(model, 0 <= y <= 3)
    JuMP.@objective(model, Min, 12x + 20y)
    JuMP.@constraint(model, c1, 6x + 8y >= 100)
    JuMP.@constraint(model, c2, 7x + 12y >= 120)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ 205.0 atol = 1.0e-2
end

if CUDA.functional()
    @info "Running CUDA tests"
    CUDA.versioninfo()

    @testset "CUDA" begin
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
    end
else
    @info "Skipping CUDA tests" CUDA.functional()
end
