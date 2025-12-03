
# Preconditioning

## Basic formulas

Original problem ``P`` and preconditioned problem ``\tilde{P}`` linked by:

- ``\tilde{A} = D_1 A D_2`` so ``A = D_1^{-1} \tilde{A} D_2^{-1}``
- ``\tilde{A}^\top = D_2 A^\top D_1`` so ``A^\top = D_2^{-1} \tilde{A}^\top D_1^{-1}``
- ``\tilde{x} = D_2^{-1} x`` so ``x = D_2 \tilde{x}``
- ``\tilde{y} = D_1^{-1} y`` so ``y = D_1 \tilde{y}``, but ``\tilde{\mathcal{Y}} = \mathcal{Y}``
- ``\tilde{r} = D_2 r`` so ``r = D_2^{-1} \tilde{r}``, but ``\tilde{\mathcal{R}} = \mathcal{R}``
- ``\tilde{c} = D_2 c`` so ``c = D_2^{-1} \tilde{c}``
- ``(\tilde{\ell}_v, \tilde{u}_v) = D_2^{-1} (\ell_v, u_v)`` so ``(\ell_v, u_v) = D_2 (\tilde{\ell}_v, \tilde{u}_v)``
- ``(\tilde{\ell}_c, \tilde{u}_c) = D_1 (\ell_c, u_c)`` so ``(\ell_c, u_c) = D_1^{-1} (\tilde{\ell}_c, \tilde{u}_c)``

## Error computation in the original problem

Then we have the following terms in the KKT errors:

```math
\begin{align*}
c - A^\top y - r
& = D_2^{-1} \tilde{c} - D_2^{-1} \tilde{A}^\top D_1^{-1} D_1 \tilde{y} - D_2^{-1} \tilde{r} \\
& = D_2^{-1}(\tilde{c} - \tilde{A}^\top \tilde{y} - \tilde{r})
\end{align*}
```

```math
\begin{align*}
Ax - \mathrm{proj}_{[\ell_c,u_c]}(Ax)
& = D_1^{-1} \tilde{A} D_2^{-1} D_2 \tilde{x} - \mathrm{proj}_{[D_1^{-1} \tilde{\ell}_c, D_1^{-1} \tilde{u}_c]} (D_1^{-1} \tilde{A} D_2^{-1} D_2 \tilde{x}) \\
& = D_1^{-1} \tilde{A} \tilde{x} - \mathrm{proj}_{[D_1^{-1} \tilde{\ell}_c, D_1^{-1} \tilde{u}_c]} (D_1^{-1} \tilde{A} \tilde{x}) \\
& = D_1^{-1} \left[\tilde{A} \tilde{x} - \mathrm{proj}_{[\tilde{\ell}_c, \tilde{u}_c]} (\tilde{A} \tilde{x})\right] \\
\end{align*}
```

```math
r - \mathrm{proj}_{\mathcal{R}}(r) = D_2^{-1} \tilde{r} - \mathrm{proj}_{\tilde{\mathcal{R}}}(D_2^{-1} \tilde{r}) = D_2^{-1} (\tilde{r} - \mathrm{proj}_{\tilde{\mathcal{R}}}(\tilde{r}))
```

```math
c^\top x = (D_2^{-1} \tilde{c})^\top (D_2 \tilde{x}) = \tilde{c}^\top D_2^{-1} D_2 \tilde{x} = \tilde{c}^\top \tilde{x}
```

```math
\begin{align*}
p(y; \ell_c, u_c)
& = u_c^\top y^+ - \ell_c^\top y^- \\
& = (D_1^{-1} \tilde{u}_c)^\top (D_1 \tilde{y})^+ - (D_1^{-1} \tilde{\ell}_c)^\top (D_1 \tilde{y})^- \\
& = \tilde{u}_c^\top D_1^{-1} D_1 \tilde{y}^+ - \tilde{\ell}_c^\top D_1^{-1} D_1 \tilde{y}^- \\
& = \tilde{u}_c^\top \tilde{y}^+ - \tilde{\ell}_c \tilde{y}^-
\end{align*}
```

```math
\begin{align*}
p(r; \ell_v, u_v)
& = u_v^\top r^+ - \ell_v^\top r^- \\
& = (D_2 \tilde{u}_v)^\top (D_2^{-1} \tilde{r})^+ - (D_2 \tilde{\ell}_v)^\top (D_2^{-1} \tilde{r})^- \\
& = \tilde{u}_v^\top D_2 D_2^{-1} \tilde{r}^+ - \tilde{\ell}_v^\top D_2 D_2^{-1} \tilde{r}^- \\
& = \tilde{u}_v^\top \tilde{r}^+ - \tilde{\ell}_v^\top \tilde{r}^-
\end{align*}
```

We make use of a few key observations:

- Projection on ``\mathcal{R}`` commutes with scaling
- Projection on an interval commutes with scaling if scaling is also applied to the interval in question
