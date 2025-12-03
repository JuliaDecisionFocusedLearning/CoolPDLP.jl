using CoolPDLP
using JLArrays
using Test

milp, sol = CoolPDLP.random_milp_and_sol(10, 20, 0.4)
params = CoolPDLP.ConversionParameters(Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend())

milp_gpu = CoolPDLP.perform_conversion(milp, params)
@test milp_gpu isa MILP{Float32, JLVector{Float32}, GPUSparseMatrixCSR{Float32, Int32, JLVector{Float32}, JLVector{Int32}}}

sol_gpu = CoolPDLP.perform_conversion(sol, params)
@test sol_gpu isa PrimalDualSolution{Float32, JLVector{Float32}}
