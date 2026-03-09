using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test

@testset "permute_rows_columns(nothing)" begin
    @test isnothing(CoolPDLP.permute_rows_columns(nothing; perm_col = [1, 2], perm_row = [1, 2]))
end

@testset "Sort columns" begin
    for m in (10, 20, 30), n in (10, 20, 30), p in (0.01, 0.1, 0.3)
        A = sprand(n, n, p)
        perm_col = CoolPDLP.increasing_column_order(A)
        perm_row = CoolPDLP.increasing_column_order(sparse(transpose(A)))
        A_sorted = CoolPDLP.permute_rows_columns(A; perm_col, perm_row)
        @test A[:, perm_col][perm_row, :] == A_sorted
        @test issorted(map(col -> count(!iszero, col), eachcol(A_sorted)))
        @test issorted(map(row -> count(!iszero, row), eachrow(A_sorted)))
    end
end

@testset "sort_rows_columns(LinearProgram)" begin
    n, m = 15, 8
    A = sprand(m, n, 0.3)
    lp = LinearProgram(;
        c = rand(n), lv = zeros(n), uv = ones(n),
        A, lc = zeros(m), uc = ones(m),
    )
    lp_sorted = CoolPDLP.sort_rows_columns(lp)
    @test lp_sorted isa LinearProgram
    @test nbvar(lp_sorted) == nbvar(lp)
    @test nbcons(lp_sorted) == nbcons(lp)
    @test lp_sorted.A != lp.A
end

@testset "sort_rows_columns(QuadraticProgram)" begin
    n, m = 15, 8
    A = sprand(m, n, 0.3)
    H = sprand(n, n, 0.3)
    Q = H' * H
    qp = QuadraticProgram(;
        c = rand(n), lv = zeros(n), uv = ones(n),
        A, Q, lc = zeros(m), uc = ones(m),
    )
    qp_sorted = CoolPDLP.sort_rows_columns(qp)
    @test qp_sorted isa QuadraticProgram
    @test nbvar(qp_sorted) == nbvar(qp)
    @test nbcons(qp_sorted) == nbcons(qp)
    @test qp_sorted.A != qp.A
    @test qp_sorted.Q != qp.Q
end
