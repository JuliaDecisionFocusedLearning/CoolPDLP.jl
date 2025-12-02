@enum Algorithm PDHG

"""
    Parameters

# Fields

$(TYPEDFIELDS)
"""
struct Parameters{
        algo,
        T <: Number,
        G <: GenericParameters{T},
        P <: PreconditioningParameters{T},
        S <: StepSizeParameters{T},
        F <: TerminationParameters{T},
    }
    generic::G
    preconditioning::P
    step_size::S
    termination::F

    function Parameters{algo}(
            generic::GenericParameters{T},
            preconditioning::PreconditioningParameters{T},
            step_size::StepSizeParameters{T},
            termination::TerminationParameters{T},
        ) where {algo, T}
        return new{
            algo,
            T,
            typeof(generic),
            typeof(preconditioning),
            typeof(step_size),
            typeof(termination),
        }(generic, preconditioning, step_size, termination)
    end
end

function Base.show(io::IO, params::Parameters{algo}) where {algo}
    (; generic, preconditioning, step_size, termination) = params
    return print(
        io, """
        Parameters for $algo:
          - $generic
          - $preconditioning
          - $step_size
          - $termination
        """
    )
end
