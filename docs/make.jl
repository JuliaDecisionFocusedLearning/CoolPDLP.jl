using CoolPDLP
using Documenter
using Literate

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force = true)

Literate.markdown(
    joinpath(@__DIR__, "..", "test", "tutorial.jl"), joinpath(@__DIR__, "src")
)

makedocs(;
    modules = [CoolPDLP],
    authors = "Guillaume Dalle",
    sitename = "CoolPDLP.jl",
    pages = ["Home" => "index.md", "tutorial.md", "api.md"],
)

deploydocs(; repo = "github.com/gdalle/CoolPDLP.jl", devbranch = "main")
