using Test
import MathOptInterface as MOI
import CoolPDLP
import JuMP

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
