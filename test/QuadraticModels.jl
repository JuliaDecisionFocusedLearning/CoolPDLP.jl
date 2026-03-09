using Test, SparseArrays, Adapt
using CoolPDLP, QuadraticModels
using CUDA, CUDA.CUSPARSE
using JLArrays
using KernelAbstractions
using GPUArraysCore: @allowscalar

c = [2.0, 3.0]
H = spzeros(2, 2)
A = sparse([1, 1], [1, 2], [1.0, 2.0], 1, 2)
lvar = [0.0, 0.0]
uvar = [5.0, 5.0]
lcon = [1.0]
ucon = [4.0]

@testset "QuadraticModel → CPU MILP" begin
    qm = QuadraticModel(c, H; A, lvar, uvar, lcon, ucon, name = "tiny_lp")
    milp = CoolPDLP.MILP(qm)

    @test milp.c ≈ c
    @test milp.lv ≈ lvar
    @test milp.uv ≈ uvar
    @test milp.lc ≈ lcon
    @test milp.uc ≈ ucon
    @test Matrix(milp.A) ≈ Matrix(A)
    @test milp.name == "tiny_lp"
end

@testset "QuadraticModel → device MILP" begin
    A_dev = adapt(JLBackend(), GPUSparseMatrixCOO(A))
    H_dev = adapt(JLBackend(), GPUSparseMatrixCOO(H))
    c_dev = jl(c)
    lv_dev = jl(lvar)
    uv_dev = jl(uvar)
    lc_dev = jl(lcon)
    uc_dev = jl(ucon)

    # we need @allowscalar since initializing NLPModelMeta uses findall
    qm = @allowscalar QuadraticModel(
        c_dev, H_dev;
        A = A_dev, lvar = lv_dev, uvar = uv_dev, lcon = lc_dev, ucon = uc_dev,
        name = "tiny_lp",
    )
    milp = CoolPDLP.MILP(qm)

    @test milp.c isa JLVector{Float64}
    @test milp.lv isa JLVector{Float64}
    @test milp.lc isa JLVector{Float64}
    @test milp.A isa GPUSparseMatrixCOO{Float64, Int, JLVector{Float64}, JLVector{Int}}
    @test milp.At isa GPUSparseMatrixCOO{Float64, Int, JLVector{Float64}, JLVector{Int}}
    @test get_backend(milp.A) isa JLBackend

    @test Array(milp.c) ≈ c
    @test Array(milp.lv) ≈ lvar
    @test Array(milp.uv) ≈ uvar
    @test Array(milp.lc) ≈ lcon
    @test Array(milp.uc) ≈ ucon
    @test milp.name == "tiny_lp"
end

if CUDA.functional()
    @testset "QuadraticModel → CUDA MILP" begin
        A_cu = CuSparseMatrixCSR(A)
        H_cu = CuSparseMatrixCSR(H)
        c_cu = CuVector(c)
        lv_cu = CuVector(lvar)
        uv_cu = CuVector(uvar)
        lc_cu = CuVector(lcon)
        uc_cu = CuVector(ucon)

        # we need @allowscalar since initializing NLPModelMeta uses findall
        qm = @allowscalar QuadraticModel(
            c_cu, H_cu;
            A = A_cu, lvar = lv_cu, uvar = uv_cu, lcon = lc_cu, ucon = uc_cu,
            name = "tiny_lp",
        )
        milp = CoolPDLP.MILP(qm)

        @test milp.c isa CuVector{Float64}
        @test milp.lv isa CuVector{Float64}
        @test milp.lc isa CuVector{Float64}
        @test milp.A isa CuSparseMatrixCSR
        @test milp.At isa CuSparseMatrixCSR

        @test Array(milp.c) ≈ c
        @test Array(milp.lv) ≈ lvar
        @test Array(milp.uv) ≈ uvar
        @test Array(milp.lc) ≈ lcon
        @test Array(milp.uc) ≈ ucon
        @test milp.name == "tiny_lp"
    end
else
    @info "Skipping CUDA QuadraticModels test" CUDA.functional()
end
