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

@testset "Symmetrized size" begin
    A = randn(10, 20)
    S = CoolPDLP.Symmetrized(A, Matrix(transpose(A)))
    @test size(S) == (20, 20)
    @test size(S, 1) == 20
    @test size(S, 2) == 20
    @test size(S, 3) == 1
    @test size(S, 100) == 1
end

@testset "Symmetrized mul! dimension mismatch" begin
    A = randn(10, 20)
    S = CoolPDLP.Symmetrized(A, Matrix(transpose(A)))
    @test_throws DimensionMismatch mul!(zeros(20), S, randn(19))
    @test_throws DimensionMismatch mul!(zeros(19), S, randn(20))
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
        s1 = CoolPDLP.spectral_norm(A, Matrix(transpose(A)); tol = 1.0e-7, maxiter = 1000)
        s1_ref = opnorm(A, 2)
        # a deterministic random start and a generous,
        # size-independent iteration budget (#95) makes this accurate to near machine
        # precision, instead of the `rtol = 1.0e-1` this used to require
        @test s1 ≈ s1_ref rtol = 1.0e-10
    end
end

@testset "Spectral norm requires an explicit tolerance and iteration budget" begin
    # neither `tol` nor `maxiter` may fall back to size-dependent (IterativeSolvers-style)
    # defaults (see #95): those scale as `eps(T) * n^3` and `size(B, 2)` respectively,
    # which become far too loose (or too small) to trust at scale
    A = randn(5, 5)
    Aᵀ = Matrix(transpose(A))
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ)
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ; tol = 1.0e-7)
    @test_throws UndefKeywordError CoolPDLP.spectral_norm(A, Aᵀ; maxiter = 1000)
end

@testset "Spectral norm of a zero matrix is exactly zero" begin
    # `Kᵀ * K` is positive semidefinite, so the largest Ritz value `λ` is
    # mathematically nonnegative; a genuinely zero spectral norm (no constraint rows, or
    # an all-zero `K`) must not be rejected by the guard against a slightly-negative `λ`
    # reaching `sqrt` (see #95 and the "Fixed step size stays finite" test)
    Z = zeros(2, 2)
    @test CoolPDLP.spectral_norm(Z, Z; tol = 1.0e-3, maxiter = 1000) == 0.0
end

@testset "Spectral norm rejects NaN entries and non-BLAS element types" begin
    A = [1.0 NaN; 0.0 2.0]
    @test_throws ArgumentError CoolPDLP.spectral_norm(A, Matrix(transpose(A)); tol = 1.0e-3, maxiter = 100)
    B = BigFloat[1 0; 0 2]
    @test_throws ArgumentError CoolPDLP.spectral_norm(B, B; tol = 1.0e-3, maxiter = 100)
    @test CoolPDLP.spectral_norm(Float32[1 0; 0 2], Float32[1 0; 0 2]; tol = 1.0e-4, maxiter = 100) ≈ 2.0f0
end

@testset "Spectral norm warns and stays conservative instead of crashing" begin
    # a `maxiter` too small for the requested `tol` used to throw an `AssertionError`
    # (traininstance2, traininstance6, triptim1 from MIPLIB 2017); it must now warn and
    # return a finite estimate, never above the rigorous bound
    rng = Xoshiro(42)
    A = randn(rng, 10, 20)
    Aᵀ = Matrix(transpose(A))
    s = @test_logs (:warn, r"did not converge") CoolPDLP.spectral_norm(A, Aᵀ; tol = 1.0e-12, maxiter = 1)
    @test 0 < s <= sqrt(opnorm(A, 1) * opnorm(A, Inf))
    # with a clustered top of the spectrum (the realistic cause of non-convergence), the
    # inflated estimate lies above the true norm, i.e. on the safe step size side
    λs = vcat(1.0, 1.0 - 1.0e-7, range(0.0, 0.999; length = 98))
    K = sparse(Diagonal(sqrt.(λs)))
    s = @test_logs (:warn, r"did not converge") CoolPDLP.spectral_norm(K, K; tol = 1.0e-12, maxiter = 30)
    @test 1.0 <= s <= 1.05
    @test CoolPDLP.spectral_norm_upper_bound(A, Aᵀ) ≈ sqrt(opnorm(A, 1) * opnorm(A, Inf))
    S = sparse(A)
    @test CoolPDLP.spectral_norm_upper_bound(S, sparse(Aᵀ)) ≈ sqrt(opnorm(A, 1) * opnorm(A, Inf))
end

@testset "Spectral norm converges where plain power iteration does not" begin
    # eigenvalues of `KᵀK`: 1, 0.995 and a bulk in [0, 0.99]; with residual tolerance 1e-6
    # the power method needs about log(1e-6 / 0.005) / log(0.995) ≈ 1700 iterations,
    # while Lanczos converges like the power method with the square root of that ratio
    n = 200
    λs = vcat(1.0, 0.995, range(0.0, 0.99; length = n - 2))
    K = sparse(Diagonal(sqrt.(λs)))
    Kᵀ = copy(transpose(K))
    tol, maxiter = 1.0e-6, 1000
    # plain power iteration with the same start and stopping rule as `powm!`
    x = randn(Xoshiro(0), n)
    x ./= norm(x)
    power_converged = false
    for _ in 1:maxiter
        y = Kᵀ * (K * x)
        θ = dot(x, y)
        if norm(y - θ * x) <= tol * θ
            power_converged = true
            break
        end
        x = y / norm(y)
    end
    @test !power_converged
    s = @test_logs CoolPDLP.spectral_norm(K, Kᵀ; tol, maxiter)
    @test s ≈ 1.0 rtol = 1.0e-8
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

@testset "Spectral norm on the CoolPDLP sparse formats" begin
    # `abs_entries` and the products in `spectral_norm` must work on every format the solver
    # converts `A` to, not only on `SparseMatrixCSC`
    rng = Xoshiro(3)
    S = sprandn(rng, 30, 40, 0.2)
    St = sparse(transpose(S))
    ref = opnorm(Matrix(S), 2)
    bound = sqrt(opnorm(Matrix(S), 1) * opnorm(Matrix(S), Inf))
    for F in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
        A, At = F(S), F(St)
        @test SparseMatrixCSC(CoolPDLP.abs_entries(A)) == abs.(S)
        @test CoolPDLP.spectral_norm_upper_bound(A, At) ≈ bound
        @test CoolPDLP.spectral_norm(A, At; tol = 1.0e-8, maxiter = 1000) ≈ ref rtol = 1.0e-6
    end
end

@testset "Spectral norm warns at every non-converging call" begin
    rng = Xoshiro(4)
    A = randn(rng, 10, 20)
    At = Matrix(transpose(A))
    for _ in 1:2
        @test_logs (:warn, r"did not converge") CoolPDLP.spectral_norm(A, At; tol = 1.0e-12, maxiter = 1)
    end
end

@testset "Upper spectral norm estimate" begin
    # the Ritz value lies below the norm; `upper = true` adds the residual bound on top of it
    rng = Xoshiro(5)
    S = sprandn(rng, 200, 300, 0.05)
    St = sparse(transpose(S))
    ref = opnorm(Matrix(S), 2)
    lo = CoolPDLP.spectral_norm(S, St; tol = 1.0e-3, maxiter = 1000)
    hi = CoolPDLP.spectral_norm(S, St; tol = 1.0e-3, maxiter = 1000, upper = true)
    @test lo <= ref * (1 + 1.0e-12)
    @test lo <= hi
    @test hi >= ref * (1 - 1.0e-12)
    @test hi <= ref * (1 + 1.0e-3)
    # the keyword of the algorithms reaches the estimate
    @test !PDLP().step_size.spectral_norm_upper
    params = PDLP(; spectral_norm_upper = true).step_size
    @test params.spectral_norm_upper
    @test CoolPDLP.spectral_norm(S, St, params) == CoolPDLP.spectral_norm(S, St; tol = params.spectral_norm_tol, maxiter = params.spectral_norm_maxiter, upper = true)
end
