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

@testset "Sort rows and columns of a MILP" begin
    milp, _ = CoolPDLP.random_milp_and_sol(10, 20, 0.3)
    milp_sorted = CoolPDLP.sort_rows_columns(milp)
    # default names are the original indices, so they tell where each row and column went
    perm_var = parse.(Int, milp_sorted.var_names)
    perm_cons = parse.(Int, milp_sorted.con_names)
    @test milp_sorted.A == milp.A[perm_cons, perm_var]
    @test milp_sorted.c == milp.c[perm_var]
    @test milp_sorted.lc == milp.lc[perm_cons]
    @test milp_sorted.uc == milp.uc[perm_cons]
end
