using Adapt
using CoolPDLP
using JLArrays
using Test

m, n = 10, 20
c = rand(n)
lv = rand(n)
uv = lv + rand(n)
A = sprand(m, n, 0.3)
lc = rand(m)
uc = lc + rand(m)
int_var = rand(Bool, length(c))
milp = MILP(; c, lv, uv, A, lc, uc, int_var)

@testset "Set types" begin
    milp_f32 = CoolPDLP.set_eltype(Float32, milp)
    @test milp_f32 isa MILP{Float32, Vector{Float32}, SparseMatrixCSC{Float32, Int}}
    milp_i32 = CoolPDLP.set_indtype(Int32, milp)
    @test milp_i32 isa MILP{Float64, Vector{Float64}, SparseMatrixCSC{Float64, Int32}}
    milp_32 = CoolPDLP.single_precision(milp)
    @test milp_32 isa MILP{Float32, Vector{Float32}, SparseMatrixCSC{Float32, Int32}}
    milp_dense = CoolPDLP.set_matrix_type(Matrix, milp)
    @test milp_dense isa MILP{Float64, Vector{Float64}, Matrix{Float64}}
end

@testset "Change backend" begin
    milp_flexible = CoolPDLP.set_matrix_type(GPUSparseMatrixCSR, milp)
    @test milp_flexible isa MILP{Float64, Vector{Float64}, GPUSparseMatrixCSR{Float64, Int, Vector{Float64}, Vector{Int}}, Vector{Bool}}
    milp_gpu = adapt(JLBackend(), milp_flexible)
    @test milp_gpu isa MILP{Float64, JLVector{Float64}, GPUSparseMatrixCSR{Float64, Int, JLVector{Float64}, JLVector{Int}}, JLVector{Bool}}
end
