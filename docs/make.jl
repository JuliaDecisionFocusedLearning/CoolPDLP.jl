using CoolPDLP
using Documenter
using Literate
using MathOptInterface

cp(
    joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md"); force = true
)

Literate.markdown(
    joinpath(@__DIR__, "..", "test", "tutorial.jl"),
    joinpath(@__DIR__, "src")
)

# DispatchDoctor's `@stable` macro (used throughout CoolPDLP) generates a
# hidden gensym'd "simulator" twin for every stabilized function (e.g.
# `##foo_simulator#123`), and that twin ends up with a copy of `foo`'s
# docstring. Strip those entries so they don't show up in the generated docs.
# See https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl/issues/92.
# Root cause upstream: https://github.com/MilesCranmer/DispatchDoctor.jl/issues/42,
# fix proposed but unmerged in https://github.com/MilesCranmer/DispatchDoctor.jl/pull/90.
# This workaround can be dropped once that lands in a released version.
let meta = Base.Docs.meta(CoolPDLP)
    for binding in collect(keys(meta))
        occursin('#', string(binding.var)) && delete!(meta, binding)
    end
end

makedocs(;
    modules = [CoolPDLP],
    authors = "Guillaume Dalle and Michael Klamkin",
    sitename = "CoolPDLP.jl",
    pages = [
        "Home" => "index.md",
        "tutorial.md",
        "api.md",
        "Dev docs" => [
            "math.md",
            "internals.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl", devbranch = "main"
)
