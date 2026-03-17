using CoolPDLP
using DispatchDoctor
using Test

@test CoolPDLP._unstable_relu(1.0) == 1.0
@test_throws DispatchDoctor.TypeInstabilityError CoolPDLP._unstable_relu(0)
