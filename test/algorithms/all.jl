using Adapt
using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using KernelAbstractions
using LinearAlgebra
using MathOptBenchmarkInstances
using JuMP: JuMP, MOI
using SparseArrays
using Test

netlib_milps = map(list_instances(Netlib)) do name
    MILP(read_instance(dataset, name)[1]; dataset, name)
end
sort!(netlib_milps, by = milp -> nbvar(milp))
small_names = filter(map(milp -> milp.name, netlib_milps[1:3])) do name
    !in(name, ["agg", "blend", "dfl001", "forplan", "gfrd-pnc", "sierra"])
end

function test_optimizer(
        dataset::MathOptBenchmarkInstances.Dataset, name::String, algo::CoolPDLP.Algorithm;
        obj_rtol::Float64 = 1.0e-2, cons_tol::Float64 = 1.0e-2, int_tol::Float64 = Inf,
    )
    qps, path = read_instance(dataset, name)
    milp = MILP(qps; dataset, path)

    jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.optimize!(jump_model)
    jump_x = JuMP.value.(JuMP.all_variables(jump_model))

    sol, stats = solve(milp, algo)
    x = sol.x

    @test stats.termination_status == CoolPDLP.OPTIMAL
    @test is_feasible(Array(x), milp; cons_tol, int_tol)
    @test isapprox(objective_value(jump_x, milp), objective_value(Array(x), milp); rtol = obj_rtol)
    return nothing
end

configs = [(SparseMatrixCSC, CPU()), (GPUSparseMatrixCSR, JLBackend())]

@testset "PDHG" begin
    @testset for (M, backend) in configs
        algo = PDHG(Float64, Int, M; backend, termination_reltol = 1.0e-1, max_kkt_passes = 10^7, show_progress = false)
        dataset = Netlib
        @testset for name in small_names
            test_optimizer(dataset, name, algo; cons_tol = 1.0e-1, obj_rtol = 1.0e-1)
        end
    end
end

@testset "PDLP" begin
    @testset for (M, backend) in configs
        algo = PDLP(Float64, Int, M; backend, termination_reltol = 1.0e-5, max_kkt_passes = 10^7, show_progress = false)
        dataset = Netlib
        @testset for name in small_names
            test_optimizer(dataset, name, algo; cons_tol = 1.0e-2)
        end
    end
end

@testset "CPU-GPU coherence" begin
    milp = netlib_milps[4]
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), termination_reltol = 1.0e-3, check_every = 1, show_progress = false)
    algo_gpu = PDLP(Float64, Int, GPUSparseMatrixCSR; backend = JLBackend(), termination_reltol = 1.0e-3, check_every = 1, show_progress = false)
    _, stats = solve(milp, algo)
    _, stats_gpu = solve(milp, algo_gpu)
    @test stats.err.primal != stats_gpu.err.primal
    @test stats.err ≈ stats_gpu.err
    @test stats.kkt_passes == stats_gpu.kkt_passes
end
