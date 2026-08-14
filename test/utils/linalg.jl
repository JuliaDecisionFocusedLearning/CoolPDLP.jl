using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test
using Random: Xoshiro

@testset "Simple projections" begin
    for x in randn(100)
        @test x ≈ CoolPDLP.positive_part(x) - CoolPDLP.negative_part(x)
        @test CoolPDLP.positive_part(x) >= 0
        @test CoolPDLP.negative_part(x) >= 0
    end
end

@testset "Projection multiplier" begin
    for y in randn(100)
        @test CoolPDLP.proj_multiplier(y, -Inf, Inf) == 0
        @test CoolPDLP.proj_multiplier(y, -Inf, 3.0) == -CoolPDLP.negative_part(y)
        @test CoolPDLP.proj_multiplier(y, -3.0, Inf) == CoolPDLP.positive_part(y)
        @test CoolPDLP.proj_multiplier(y, -3.0, 3.0) == y
    end
end

@testset "Safe product" begin
    # normal (finite `left`) case is an ordinary product
    @test CoolPDLP.safeprod_left(3.0, 5.0) == 15.0
    @test CoolPDLP.safeprod_left(-3.0, 5.0) == -15.0

    # well-behaved case: paired multiplier is exactly zero whenever the bound is infinite
    @test CoolPDLP.safeprod_left(Inf, 0.0) == 0.0
    @test CoolPDLP.safeprod_left(-Inf, 0.0) == 0.0

    # invariant-violating case (e.g. a bad user-supplied warm start): an infinite bound
    # must still zero out the term instead of leaking the raw multiplier value. This has
    # to hold even for a nonzero `right`, since a legitimately converging PDHG iterate can
    # leave tiny nonzero floating-point residuals on the multiplier of an unconstrained row
    # (a `σ`/`inv(σ)` round-trip is not bit-exact), and those residuals must not blow up
    # into `±Inf`/`NaN` once multiplied by an infinite bound and summed across rows
    @test CoolPDLP.safeprod_left(Inf, 5.0) == 0.0
    @test CoolPDLP.safeprod_left(-Inf, 5.0) == 0.0
end

@testset "Bound scale" begin
    @test CoolPDLP.combine(1, 2) == 2
    @test CoolPDLP.combine(3, 3) == 3
    @test CoolPDLP.combine(-Inf, 2) == 2
    @test CoolPDLP.combine(3, Inf) == 3
    @test CoolPDLP.combine(-Inf, Inf) == 0
end

@testset "Symmetrized" begin
    for _ in 1:10
        A = randn(10, 20)
        S = CoolPDLP.Symmetrized(A, Matrix(transpose(A)))
        x = randn(20)
        y = zeros(20)
        mul!(y, S, x)
        @test y ≈ transpose(A) * A * x
    end
end

@testset "Symmetrized with mismatched K/Kᵀ types" begin
    # K and Kᵀ need not share a concrete matrix type (see issue #102)
    A = randn(10, 20)
    S = CoolPDLP.Symmetrized(A, sparse(Matrix(transpose(A))))
    x = randn(20)
    y = zeros(20)
    mul!(y, S, x)
    @test y ≈ transpose(A) * A * x
end

@testset "Spectral norm" begin
    rng = Xoshiro(42)
    for _ in 1:10
        A = randn(rng, 10, 20)
        s1 = CoolPDLP.spectral_norm(A, Matrix(transpose(A)); tol = 1.0e-7)
        s1_ref = opnorm(A, 2)
        @test s1 ≈ s1_ref rtol = 1.0e-1
    end
end

@testset "Column reductions" begin
    v, m = randn(4), randn(4, 3)

    @test CoolPDLP.colnorm!!(0.0, v) ≈ norm(v)
    @test CoolPDLP.colsum!!(0.0, v) ≈ sum(v)

    dest = zeros(3)
    @test CoolPDLP.colnorm!!(dest, m) === dest
    @test dest ≈ norm.(eachcol(m))
    @test CoolPDLP.colsum!!(dest, m) === dest
    @test dest ≈ sum.(eachcol(m))
end

@testset "Nonzero count" begin
    A = sprand(10, 8, 0.4)
    @test CoolPDLP.mynnz(A) == nnz(A)
    @test CoolPDLP.mynnz(Matrix(A)) == 80
end

@testset "Column norm" begin
    A = sprand(10, 10, 0.6)
    for j in axes(A, 2)
        for p in (0.1, 1.0, 2.0)
            @test CoolPDLP.column_norm(A, j, p) ≈ norm(A[:, j], p)
        end
    end
end

@testset "Same-type transpose" begin
    for A in Any[rand(10, 10), sprand(10, 10, 0.6)]
        @test CoolPDLP.sametype_transpose(A) == transpose(A)
        @test CoolPDLP.sametype_transpose(A) isa typeof(A)
    end
end
