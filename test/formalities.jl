using Aqua
using CoolPDLP
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(CoolPDLP; undocumented_names = true)
end;

@testset "JET" begin
    JET.test_package(CoolPDLP)
end
