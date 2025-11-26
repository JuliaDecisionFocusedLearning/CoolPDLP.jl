"""
    AlgorithmParameters

# Fields

$(TYPEDFIELDS)
"""
struct AlgorithmParameters{
        T <: Number,
        G <: GenericParameters{T},
        P <: PreconditioningParameters{T},
        S <: StepSizeParameters{T},
        F <: TerminationParameters{T},
    }
    algo::Symbol
    generic::G
    preconditioning::P
    step_size::S
    termination::F
end

function Base.show(io::IO, params::AlgorithmParameters)
    (; algo, generic, preconditioning, step_size, termination) = params
    return print(
        io, """
        AlgorithmParameters for $algo:
        - $generic
        - $preconditioning
        - $step_size
        - $termination
        """
    )
end
