using Aqua
using CoolPDLP
using ExplicitImports
using Test

@testset "Aqua" begin
    Aqua.test_all(CoolPDLP; undocumented_names = true)
end;

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(CoolPDLP) === nothing
    @test check_no_stale_explicit_imports(CoolPDLP) === nothing
    @test check_all_explicit_imports_via_owners(CoolPDLP) === nothing
    @test_broken check_all_explicit_imports_are_public(CoolPDLP; ignore = (QPSReader,)) === nothing
    @test check_all_qualified_accesses_via_owners(CoolPDLP) === nothing
    @test_broken check_all_qualified_accesses_are_public(CoolPDLP) === nothing
    @test check_no_self_qualified_accesses(CoolPDLP) === nothing
end
