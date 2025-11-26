using Aqua
using CoolPDLP
using Test

@testset "Aqua" begin
    Aqua.test_all(CoolPDLP; undocumented_names = true)
end;
