using CoolPDLP
using Test

iteration = CoolPDLP.IterationCounter(3, 6, 12)
CoolPDLP.add_inner!(iteration)
@test iteration.outer == 3
@test iteration.inner == 7
@test iteration.total == 13
CoolPDLP.add_outer!(iteration)
@test iteration.outer == 4
@test iteration.inner == 0
@test iteration.total == 13
