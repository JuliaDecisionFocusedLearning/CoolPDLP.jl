using Aqua
using CoolPDLP
using ExplicitImports
using Test

@testset "Aqua" begin
    Aqua.test_all(CoolPDLP; undocumented_names = true)
end;

@testset "ExplicitImports" begin
    ExplicitImports.test_explicit_imports(
        CoolPDLP;
        all_explicit_imports_are_public = false,
        all_qualified_accesses_are_public = false,
    )
end
