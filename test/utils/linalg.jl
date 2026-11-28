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

@testset "Spectral norm" begin
    rng = Xoshiro(42)
    for _ in 1:10
        A = randn(rng, 10, 20)
        s1 = CoolPDLP.spectral_norm(A, Matrix(transpose(A)); reltol = 1.0e-7, maxiter = 1000)
        s1_ref = opnorm(A, 2)
        # normalizing the initial guess and giving the power method a generous,
        # size-independent iteration budget (#95) makes this accurate to near machine
        # precision, instead of the `rtol = 1.0e-1` this used to require
        @test s1 ≈ s1_ref rtol = 1.0e-10
    end
end

@testset "Spectral norm requires an explicit tolerance and iteration budget" begin
    # neither `tol` nor `maxiter` may fall back to IterativeSolvers' size-dependent
    # defaults (see #95): those scale as `eps(T) * n^3` and `size(B, 2)` respectively,
    # which become far too loose (or too small) to trust at scale
    A = randn(5, 5)
    Aᵀ = Matrix(transpose(A))
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ)
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ; reltol = 1.0e-7)
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ; maxiter = 1000)
end

@testset "Spectral norm of a zero matrix is exactly zero" begin
    # `Kᵀ * K` is positive semidefinite, so the power method's Rayleigh quotient `λ` is
    # mathematically nonnegative; a genuinely zero spectral norm (no constraint rows, or
    # an all-zero `K`) must not be rejected by the guard against a slightly-negative `λ`
    # reaching `sqrt` (see #95 and the "Fixed step size stays finite" test)
    Z = zeros(2, 2)
    @test CoolPDLP.spectral_norm(Z, Z; reltol = 1.0e-3, maxiter = 1000) == 0.0
end

@testset "Spectral norm of a matrix with a small norm" begin
    # the relative tolerance does not depend on the scale, unlike the old absolute one
    A = randn(Xoshiro(0), 30, 60)
    for scale in (1.0e-3, 1.0e-6)
        s = CoolPDLP.spectral_norm(scale * A, scale * Matrix(transpose(A)); reltol = 1.0e-3, maxiter = 1000)
        @test s ≈ scale * opnorm(A, 2) rtol = 2.0e-3
    end
end

@testset "Spectral norm warns and stays conservative instead of crashing" begin
    # a clustered top of the spectrum, which 30 iterations cannot resolve to `reltol`
    λs = vcat(1.0, 1.0 - 1.0e-7, range(0.0, 0.999; length = 98))
    K = sparse(Diagonal(sqrt.(λs)))
    s = @test_logs (:warn, r"did not converge") CoolPDLP.spectral_norm(K, K; reltol = 1.0e-12, maxiter = 30)
    @test 1.0 <= s <= 1.05
end

@testset "Spectral norm on the CoolPDLP sparse formats, on CPU and JLArrays" begin
    using JLArrays: JLArrays, JLBackend
    JLArrays.allowscalar(false)
    milp, _ = CoolPDLP.random_milp_and_sol(60, 80, 0.1)
    ref = opnorm(Matrix(milp.A), 2)
    for F in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL), backend in (CoolPDLP.CPU(), JLBackend())
        # Krylov.jl multiplies column views, which must reach the sparse kernels
        m = CoolPDLP.perform_conversion(milp, CoolPDLP.ConversionParameters(Float32, Int32, F; backend))
        s = CoolPDLP.spectral_norm(m.A, m.At; reltol = 1.0f-5, maxiter = 1000)
        @test s isa Float32
        @test s ≈ ref rtol = 1.0e-4
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

@testset "Spectral norm in other precisions" begin
    S = sprandn(Xoshiro(6), 30, 40, 0.2)
    ref = opnorm(Matrix(S), 2)
    A16 = convert(SparseMatrixCSC{Float16, Int}, S)
    A = convert(SparseMatrixCSC{BigFloat, Int}, S)
    # precisions outside LAPACK need GenericLinearAlgebra: loaded last, since it extends LinearAlgebra
    @test_throws "load GenericLinearAlgebra" CoolPDLP.spectral_norm(A16, sparse(transpose(A16)); reltol = 1.0e-2, maxiter = 1000)
    @test_throws "load GenericLinearAlgebra" CoolPDLP.spectral_norm(A, sparse(transpose(A)); reltol = 1.0e-2, maxiter = 1000)
    @eval using GenericLinearAlgebra: GenericLinearAlgebra
    s = Base.invokelatest(CoolPDLP.spectral_norm, A16, sparse(transpose(A16)); reltol = 1.0e-2, maxiter = 1000)
    @test s isa Float16
    @test s ≈ ref rtol = 1.0e-2
    s = Base.invokelatest(CoolPDLP.spectral_norm, A, sparse(transpose(A)); reltol = BigFloat(1.0e-40), maxiter = 1000)
    @test s isa BigFloat
    @test s ≈ maximum(GenericLinearAlgebra.svdvals!(Matrix(A))) rtol = 1.0e-35
end
