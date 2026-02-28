# CoolPDLP.jl

[![tests](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![docs:stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/CoolPDLP.jl/stable)
[![docs:dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/CoolPDLP.jl/dev)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A pure-Julia, hardware-agnostic parallel implementation of Primal-Dual hybrid gradient for Linear Programming (PDLP) and its variants.

_This package is a work in progress, with many features still missing. Please reach out if it doesn't work to your satisfaction._

## Getting started

Use Julia's package manager to install `CoolPDLP.jl`, choosing either the latest stable version

```julia
pkg> add CoolPDLP
```

or the development version

```julia
pkg> add https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl
```

There are two ways to call the solver: either directly or via its [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) interface.

## Use with JuMP

To use `CoolPDLP` with JuMP, use `CoolPDLP.Optimizer`:

```julia
using CoolPDLP, JuMP

model = Model(CoolPDLP.Optimizer)
# Set `matrix_type` and `backend` to use GPU:
set_attribute(model, "matrix_type", CUSPARSE.CuSparseMatrixCSR)
set_attribute(model, "backend", CUDABackend())
```

## Why a new package?

There are already several open-source implementations of primal-dual algorithms for LPs (not to mention those in commercial solvers).
Here is an incomplete list:

| Package | Hardware |
| --- | --- |
| [`FirstOrderLP.jl`](https://github.com/google-research/FirstOrderLp.jl), [`or-tools`](https://github.com/google/or-tools) | CPU only |
| [`cuPDLP.jl`](https://github.com/jinwen-yang/cuPDLP.jl), [`cuPDLP-c`](https://github.com/COPT-Public/cuPDLP-C) | NVIDIA |
| [`cuPDLPx`](https://github.com/MIT-Lu-Lab/cuPDLPx), [`cuPDLPx.jl`](https://github.com/MIT-Lu-Lab/cuPDLPx.jl) | NVIDIA |
| [`HPR-LP`](https://github.com/PolyU-IOR/HPR-LP), [`HP-LP-C`](https://github.com/PolyU-IOR/HPR-LP-C), [`HPR-LP-PYTHON`](https://github.com/PolyU-IOR/HPR-LP-Python) | NVIDIA |
| [`BatchPDLP.jl`](https://github.com/PSORLab/BatchPDLP.jl) | NVIDIA |
| [`HiGHS`](https://github.com/ERGO-Code/HiGHS) | NVIDIA |
| [`cuopt`](https://github.com/NVIDIA/cuopt) | NVIDIA |
| [`torchPDLP`](https://github.com/SimplySnap/torchPDLP/) | agnostic (via `PyTorch`) |
| [`MPAX`](https://github.com/MIT-Lu-Lab/MPAX) | agnostic (via `JAX`) |

Unlike `cuPDLP` and most of its variants, `CoolPDLP.jl` uses [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) to target most common GPU architectures (NVIDIA, AMD, Intel, Apple), as well as plain CPUs.
It also allows you to plug in your own sparse matrix types, or experiment with different floating point precisions.
That's what makes it so cool.

## References

> [PDLP: A Practical First-Order Method for Large-Scale Linear Programming](https://arxiv.org/abs/2501.07018), Applegate et al. (2025)

> [An Overview of GPU-based First-Order Methods for Linear Programming and Extensions](https://arxiv.org/abs/2506.02174v1), Lu & Yang (2025)

## Roadmap

See the [issue tracker](https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl/issues) for an overview of planned features.

## Acknowledgements

Guillaume Dalle was partially funded through a state grant managed by Agence Nationale de la Recherche for France 2030 (grant number ANR-24-PEMO-0001).
