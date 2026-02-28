using Test
import MathOptInterface as MOI
import CoolPDLP
using CUDA
using CUDA.CUSPARSE
import JuMP
using JLArrays: JLBackend

if get(ENV, "COOLPDLP_TEST_MOISUITE", "0") == "1"
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
            ],
        );
        exclude = [
            # TODO: infeasible/unbounded detection
            r"INFEASIBILITY_CERTIFICATE", r"INFEASIBLE",  # exclude infeasible test problems
            r"test_linear_add_constraints",  # for `y`, we get 36.434290756682884 but answer is 36.36363636363637
        ],
    )
end
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

@testset "Float32" begin
    # model/return in Float64, solve in Float32
    model = JuMP.Model(CoolPDLP.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "float_type", Float32)
    JuMP.@variable(model, x >= 0)
    JuMP.@variable(model, 0 <= y <= 3)
    JuMP.@objective(model, Min, 12x + 20y)
    JuMP.@constraint(model, c1, 6x + 8y >= 100)
    JuMP.@constraint(model, c2, 7x + 12y >= 120)
    @test_warn "Got mismatched float type" JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ 205.0 atol = 1.0e-2
    @test JuMP.value(x) isa Float64

    # model/return in Float32, solve in Float32
    model = JuMP.GenericModel{Float32}(CoolPDLP.Optimizer{Float32})
    JuMP.set_silent(model)
    JuMP.@variable(model, x >= 0.0f0)
    JuMP.@variable(model, 0.0f0 <= y <= 3.0f0)
    JuMP.@objective(model, Min, 12.0f0x + 20.0f0y)
    JuMP.@constraint(model, c1, 6.0f0x + 8.0f0y >= 100.0f0)
    JuMP.@constraint(model, c2, 7.0f0x + 12.0f0y >= 120.0f0)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ 205.0f0 atol = 1.0e-2
    @test JuMP.value(x) isa Float32
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
