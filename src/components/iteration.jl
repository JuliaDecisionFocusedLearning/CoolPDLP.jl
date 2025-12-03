mutable struct IterationCounter
    outer::Int
    inner::Int
    total::Int
end

function add_inner!(iteration::IterationCounter)
    iteration.inner += 1
    iteration.total += 1
    return nothing
end

function add_outer!(iteration::IterationCounter)
    iteration.outer += 1
    iteration.inner = 0
    return nothing
end
