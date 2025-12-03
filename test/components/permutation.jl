using CoolPDLP
using SparseArrays
using Test

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
