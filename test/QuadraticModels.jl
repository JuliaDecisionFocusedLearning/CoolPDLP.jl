using Test, SparseArrays
using CoolPDLP, QuadraticModels

@testset "QuadraticModel → MILP conversion" begin
    c  = [2.0, 3.0]
    H  = spzeros(2, 2)
    A  = sparse([1, 1], [1, 2], [1.0, 2.0], 1, 2)
    lvar = [0.0, 0.0]
    uvar = [5.0, 5.0]
    lcon = [1.0]
    ucon = [4.0]

    qm = QuadraticModel(c, H; A, lvar, uvar, lcon, ucon, name = "tiny_lp")

    milp = CoolPDLP.MILP(qm)

    @test milp.c  ≈ c
    @test milp.lv ≈ lvar
    @test milp.uv ≈ uvar
    @test milp.lc ≈ lcon
    @test milp.uc ≈ ucon
    @test Matrix(milp.A) ≈ Matrix(A)
    @test milp.name == "tiny_lp"
end
