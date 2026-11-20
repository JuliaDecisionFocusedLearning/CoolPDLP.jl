using CoolPDLP
using CUDA
using cuSPARSE
using KernelAbstractions: get_backend
using Random
using Test

# a GPU matrix type moves the whole problem to the GPU even without an explicit backend, and
# the starting point used to stay on the CPU, so that `initialize` failed with scalar indexing
milp0, sol0 = CoolPDLP.random_milp_and_sol(Xoshiro(0), 20, 30, 0.4)
make_algo(backend) = PDLP(
    Float64, Int32, CuSparseMatrixCSR; backend, max_kkt_passes = 200, show_progress = false
)

milp, sol = preprocess(milp0, sol0, make_algo(nothing))
@test milp.A isa CuSparseMatrixCSR
@test sol.x isa CuArray
@test sol.y isa CuArray
@test get_backend(sol.x) == get_backend(milp)

sol_nothing, stats_nothing = solve(milp0, sol0, make_algo(nothing))
sol_cuda, stats_cuda = solve(milp0, sol0, make_algo(CUDABackend()))
@test stats_nothing.kkt_passes == stats_cuda.kkt_passes
@test Array(sol_nothing.x) ≈ Array(sol_cuda.x)
@test Array(sol_nothing.y) ≈ Array(sol_cuda.y)
