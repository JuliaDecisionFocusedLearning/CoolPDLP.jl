using Documenter
using CoolPDLP

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force = true)

makedocs(;
    modules = [CoolPDLP],
    authors = "Guillaume Dalle",
    sitename = "CoolPDLP.jl",
    pages = [
        "Home" => "index.md",
        "api.md",
    ],
)

deploydocs(;
    repo = "github.com/gdalle/CoolPDLP.jl", devbranch = "main"
)
