# CoolPDLP.jl

[![Build Status](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/CoolPDLP.jl/actions/workflows/Test.yml?query=branch%3Amain)

A pure-Julia, hardware-agnostic parallel implementation of Primal-Dual hybrid gradient for Linear Programming (PDLP).

Unlike cuPDLP and its variants, this code is designed to run on most common GPU architectures (NVIDIA, AMD, Intel, Apple).

## References

- [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://arxiv.org/abs/2106.04756), Applegate et al. (2022)
- [cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia](https://arxiv.org/abs/2311.12180), Lu et al. (2024)
- [cuPDLP-C: A Strengthened Implementation of cuPDLP for Linear Programming by C language](https://arxiv.org/abs/2312.14832), Lu et al. (2024)
- [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051), Lu et al. (2025)
