using SparseArrays
using CoolPDLP
using Test

netlib = list_netlib_instances()
milp, path = read_netlib_instance(netlib[4])

sad = SaddlePointProblem(milp)
sad_precond = precondition_pdlp(sad)

K = sad.K
K̃, D1, D2 = sad_precond.K, sad_precond.D1, sad_precond.D2
@test D1 * K * D2 ≈ K̃
