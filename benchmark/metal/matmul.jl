using Pkg
Pkg.activate(@__DIR__)

using Adapt
using CairoMakie
using Chairmarks
using CoolPDLP
using DataFrames
using LinearAlgebra
using Metal
using Random
using SparseArrays

function kkt_pass_timing(sad::SaddlePointProblem)
    (; c, q, K, Kᵀ) = sad
    x = similar(c)
    y = similar(q)
    rand!(x)
    Metal.@sync mul!(y, K, x)
    res1 = @b Metal.@sync mul!(y, K, x)
    rand!(y)
    Metal.@sync mul!(y, K, x)
    res2 = @b Metal.@sync mul!(x, Kᵀ, y)
    return res1.time + res2.time
end

nnz_K = Int[]
times32 = Float64[]
times32_device = Float64[]
times32_metal = Float64[]

instance_folder = joinpath(dirname(dirname(@__DIR__)), "challenge", "data", "instances")
for i in 1:50
    @info "$i"
    i_str = i < 10 ? "0$i" : "$i"
    milp = read_milp(joinpath(instance_folder, "instance_$i_str.mps"))
    sad = SaddlePointProblem(milp)
    sad32 = single_precision(sad)
    sad32_device = to_device(sad32)
    sad32_metal = adapt(MetalBackend(), sad32_device)

    t32 = kkt_pass_timing(sad32)
    t32_device = kkt_pass_timing(sad32_device)
    t32_metal = kkt_pass_timing(sad32_metal)

    push!(nnz_K, nnz(sad.K))
    push!(times32, t32)
    push!(times32_device, t32_device)
    push!(times32_metal, t32_metal)
end

let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "Nb of non-zeros in the KKT matrix",
        ylabel = "Duration of 1 KKT pass (seconds)",
        title = "Benchmark of sparse formats for matrix-vector products",
        subtitle = "MIPcc26 instance set",
        xscale = log10,
        yscale = log10
    )
    scatter!(ax, nnz_K, times32, marker = :cross, label = "CSC - CPU - 32 bits")
    scatter!(ax, nnz_K, times32_device, marker = :circle, label = "CSR - CPU - 32 bits")
    scatter!(ax, nnz_K, times32_metal, marker = :rect, label = "CSR - Metal GPU - 32 bits")
    axislegend(position = :lt)
    fig
end
