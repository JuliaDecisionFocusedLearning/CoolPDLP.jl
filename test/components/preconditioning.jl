using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test

begin
    m, n = 10, 20

    c = rand(n)
    lv = rand(n)
    uv = lv + rand(n)
    A = sprand(m, n, 0.3)
    At = sparse(transpose(A))
    lc = rand(m)
    uc = lc + rand(m)
    D1 = Diagonal(ones(m))
    D2 = Diagonal(ones(n))
    int_var = rand(Bool, n)
    var_names = map(string, 1:n)
    dataset = "dataset"
    name = "name"
    path = "path"

    milp = MILP(;
        c, lv, uv, A, At, lc, uc, D1, D2,
        int_var, var_names, dataset, name, path
    )
    x = randn(n)
    y = randn(m)
end

params = CoolPDLP.PreconditioningParameters(; chambolle_pock_alpha = 1, ruiz_iter = 10)
p = CoolPDLP.pdlp_preconditioner(milp, params)

milp_precond = CoolPDLP.precondition_problem(milp, p)
milp_unprecond = CoolPDLP.precondition_problem(milp_precond, inv(p))
isapprox(milp, milp_unprecond)

x_precond, y_precond = CoolPDLP.precondition_variables(x, y, p)
x_unprecond, y_unprecond = CoolPDLP.precondition_variables(x_precond, y_precond, inv(p))
@test isapprox(x, x_unprecond)
@test isapprox(y, y_unprecond)

@test (inv(p) * p).D1 ≈ I
@test (inv(p) * p).D2 ≈ I
@test (p * inv(p)).D1 ≈ I
@test (p * inv(p)).D2 ≈ I
