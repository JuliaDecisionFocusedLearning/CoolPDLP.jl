using CoolPDLP
using SparseArrays
using Test

@testset "Sort columns" begin
    for m in (10, 20, 30), n in (10, 20, 30), p in (0.01, 0.1, 0.3)
        A = sprand(n, n, p)
        perm = CoolPDLP.increasing_column_order(A)
        A_sorted, perm = CoolPDLP.permute_columns(A, perm)
        @test A[:, perm] == A_sorted
    end
end
