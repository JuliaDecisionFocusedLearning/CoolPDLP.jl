using CoolPDLP
using CoolPDLP: ConversionParameters, GPUSparseMatrixCSR, milp_to_mps, mps_to_milp, perform_conversion
using JLArrays: JLBackend
using JuMP: JuMP
using MathOptBenchmarkInstances
using MathOptInterface: MathOptInterface as MOI
using SparseArrays
using Test

@testset "MPS round trip: mixed bounds and integrality" begin
    c = [1.0, 2.0, -1.0, 0.0]
    lv = [0.0, -Inf, -3.0, -Inf]
    uv = [4.0, Inf, 3.0, Inf]
    A = sparse(
        [
            1.0 1.0 0.0 0.0
            0.0 1.0 1.0 0.0
            1.0 0.0 0.0 1.0
            0.0 1.0 0.0 1.0
        ]
    )
    # a ranged row, two one-sided rows and an equality row
    lc = [-1.0, -Inf, -3.0, 2.0]
    uc = [1.0, 5.0, Inf, 2.0]
    int_var = [false, false, true, false]
    var_names = ["alpha", "beta", "gamma", "delta"]
    con_names = ["ranged", "upper", "lower", "equal"]
    milp = MILP(; c, c0 = 2.5, lv, uv, A, lc, uc, int_var, var_names, con_names)

    path = tempname() * ".mps"
    milp_to_mps(milp, path)
    milp2 = mps_to_milp(path)

    # rows and columns keep their order, so everything can be compared element-wise
    @test milp2.var_names == milp.var_names
    @test milp2.con_names == milp.con_names
    @test milp2.c == milp.c
    @test milp2.c0 == milp.c0
    @test milp2.lv == milp.lv
    @test milp2.uv == milp.uv
    @test milp2.int_var == milp.int_var
    @test milp2.A == milp.A
    @test milp2.lc == milp.lc
    @test milp2.uc == milp.uc

    # the names to write can be overridden
    milp_to_mps(milp, path; var_names = ["x1", "x2", "x3", "x4"], con_names = ["c1", "c2", "c3", "c4"])
    milp3 = mps_to_milp(path)
    rm(path; force = true)
    @test milp3.var_names == ["x1", "x2", "x3", "x4"]
    @test milp3.con_names == ["c1", "c2", "c3", "c4"]
    @test milp3.A == milp.A
end

@testset "MPS round trip: a maximization problem is written as its minimization form" begin
    model = JuMP.Model()
    JuMP.@variable(model, 0 <= x[1:2] <= 1)
    JuMP.@constraint(model, x[1] + x[2] <= 1.5)
    JuMP.@objective(model, Max, x[1] + 2x[2] + 3)
    path_max = tempname() * ".mps"
    JuMP.write_to_file(model, path_max; format = MOI.FileFormats.FORMAT_MPS)
    milp = mps_to_milp(path_max)
    @test milp.c == [-1.0, -2.0]
    @test milp.c0 == -3.0

    path = tempname() * ".mps"
    milp_to_mps(milp, path)
    milp2 = mps_to_milp(path)
    @test milp2 ≈ milp

    # the file holds `min -cᵀx - c0`, not the original `max cᵀx + c0`
    model2 = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    @test JuMP.objective_sense(model2) == MOI.MIN_SENSE
    x2 = JuMP.all_variables(model2)
    @test JuMP.objective_function(model2) == -x2[1] - 2x2[2] - 3
    rm(path_max; force = true)
    rm(path; force = true)
end

@testset "MPS round trip: fixed variable, free variable, free row" begin
    c = [1.0, -1.0, 2.0]
    lv = [2.0, -Inf, -Inf]
    uv = [2.0, Inf, Inf]
    A = sparse([1.0 1.0 0.0; 0.0 0.0 1.0])
    lc = [-Inf, 0.0]
    uc = [Inf, 0.0]
    milp = MILP(; c, lv, uv, A, lc, uc)

    path = tempname() * ".mps"
    milp_to_mps(milp, path)
    milp2 = mps_to_milp(path)
    rm(path; force = true)

    @test milp2.var_names == milp.var_names
    @test milp2.lv == milp.lv  # in particular, the fixed and free variables keep their bounds
    @test milp2.uv == milp.uv
    # the free row constrains nothing, and does not survive the round trip
    @test milp2.con_names == milp.con_names[2:2]
    @test milp2.A == milp.A[2:2, :]
end

@testset "MPS round trip: a Float32 GPU MILP comes back as a host Float64 MILP" begin
    # MPS is a host, `Float64` format, so `milp_to_mps` must accept a MILP living anywhere and
    # `mps_to_milp` always hands back a host problem — `solve` converts it afterwards anyway
    qps, path_afiro = read_instance(Netlib, "afiro")
    milp_cpu = MILP(qps; path = path_afiro, name = "afiro")
    gpu_conv = ConversionParameters(Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend())
    milp_gpu = perform_conversion(milp_cpu, gpu_conv)

    path = tempname() * ".mps"
    milp_to_mps(milp_gpu, path)
    milp2 = mps_to_milp(path)
    rm(path; force = true)

    @test typeof(milp2) === typeof(milp_cpu)
    @test nbvar(milp2) == nbvar(milp_gpu)
    @test nbcons(milp2) == nbcons(milp_gpu)
    @test milp2.c ≈ Float64.(Array(milp_gpu.c))
    @test milp2.lv ≈ Float64.(Array(milp_gpu.lv))
    @test milp2.uv ≈ Float64.(Array(milp_gpu.uv))
end

@testset "MPS conversion rejects batched MILPs and non-linear models" begin
    milp, _ = CoolPDLP.random_milp_and_sol(4, 6, 0.5)
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    @test_throws ArgumentError milp_to_mps(milp_batch, tempname() * ".mps")

    quad_model = JuMP.Model()
    JuMP.@variable(quad_model, 0 <= q[1:2] <= 1)
    JuMP.@constraint(quad_model, q[1] + q[2] <= 1)
    JuMP.@objective(quad_model, Min, q[1] * q[2])
    quad_path = tempname() * ".mps"
    JuMP.write_to_file(quad_model, quad_path; format = MOI.FileFormats.FORMAT_MPS)
    @test_throws ArgumentError mps_to_milp(quad_path)
    rm(quad_path; force = true)
end
