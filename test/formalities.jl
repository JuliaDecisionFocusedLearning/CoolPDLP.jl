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

@testset "DispatchDoctor reaches the extensions" begin
    # an extension has its own UUID, so the CoolPDLP preference only reaches its `@stable`
    # through `CoolPDLP.dispatch_doctor_mode()`: in error mode, every method it wraps leaves a
    # `_simulator` function behind, and in disable mode none does
    using JLArrays: JLArrays
    ext = Base.get_extension(CoolPDLP, :CoolPDLPGPUArraysExt)
    @test ext isa Module
    wrapped = any(n -> occursin("_simulator", string(n)), names(ext; all = true))
    @test wrapped == (CoolPDLP.dispatch_doctor_mode() != "disable")
end
