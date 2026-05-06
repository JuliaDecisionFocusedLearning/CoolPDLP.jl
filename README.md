# CoolPDLP.jl

[![tests](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Build status](https://badge.buildkite.com/abafb7c3f7e1fd1ab4672581e288ca9dd330120e2a66851ae9.svg?branch=main)](https://buildkite.com/julialang/coolpdlp-dot-jl)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/CoolPDLP.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/CoolPDLP.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![docs:stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/CoolPDLP.jl/stable)
[![docs:dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/CoolPDLP.jl/dev)
[![DOI](https://zenodo.org/badge/1086418051.svg)](https://doi.org/10.5281/zenodo.19064770)

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

To use `CoolPDLP` with JuMP, select `CoolPDLP.Optimizer` and customize the options:

```julia
using CoolPDLP, JuMP, CUDA, cuSPARSE

model = Model(CoolPDLP.Optimizer)
# Set `matrix_type` and `backend` to use GPU:
set_attribute(model, "matrix_type", cuSPARSE.CuSparseMatrixCSR)
set_attribute(model, "backend", CUDABackend())
# Build and solve model as usual
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

- [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://arxiv.org/abs/2106.04756), Applegate et al. (2022)
- [cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia](https://arxiv.org/abs/2311.12180), Lu et al. (2024)
- [cuPDLP-C: A Strengthened Implementation of cuPDLP for Linear Programming by C language](https://arxiv.org/abs/2312.14832), Lu et al. (2024)
- [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051), Lu et al. (2025)
- [PDLP: A Practical First-Order Method for Large-Scale Linear Programming](https://arxiv.org/abs/2501.07018), Applegate et al. (2025)
- [An Overview of GPU-based First-Order Methods for Linear Programming and Extensions](https://arxiv.org/abs/2506.02174v1), Lu & Yang (2025)

## Roadmap

See the [issue tracker](https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl/issues) for an overview of planned features.

## Acknowledgements

Guillaume Dalle was partially funded through a state grant managed by Agence Nationale de la Recherche for France 2030 (grant number ANR-24-PEMO-0001).

This material is based upon work supported by the National Science Foundation AI Institute for Advances in Optimization ([AI4OPT](https://ai4opt.org)) under Grant No. 2112533 and the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2039655. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
