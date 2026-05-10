# Math

> See the references in the README for details, but beware that we adopt slightly different notational conventions.

## Primal and dual problems

PDLP solves Linear Programs (LPs) formulated as follows:

```math
\min_x \quad c^\top x \quad \text{s.t.} \quad \begin{cases}
\ell_c \leq A x \leq u_c \\
\ell_v \leq x \leq u_v
\end{cases}
```

We associate non-negative multipliers $y_\ell, y_u, z_\ell, z_u \geq 0$ with all four inequality constraints, leading to the following Lagrangian:

```math
\begin{align*}
\mathcal{L}(x, y_\ell, y_u, z_\ell, z_u)
& = c^\top x + y_\ell^\top (\ell_c - A x) + y_u^\top (A x - u_c) + z_\ell^\top (\ell_v - x) + z_u^\top (x - u_v) \\
& = (c - A^\top y_\ell + A^\top y_u - z_\ell + z_u)^\top x + (y_\ell^\top \ell_c - y_u^\top u_c) + (z_\ell^\top \ell_v - z_u^\top u_v)
\end{align*}
```

We interpret signed multipliers $y_\ell, z_\ell$ and $y_u, z_u$ as the positive and negative parts of unsigned multipliers $y$ and $z$, associated with the constraints and the variable bounds respectively:

```math
y = y_\ell - y_u \quad \text{and} \quad z = z_\ell - z_u
```

which amounts to

```math
\begin{align*}
y_\ell & = y^+ & z_\ell & = z^+ \\
y_u & = y^- & z_u & = z^-
\end{align*}
```

Note that if any of the bounds is infinite, the corresponding signed multiplier is constrained to be zero.
We sum up these elementwise constraints by writing $y \in \mathcal{Y}$ and $z \in \mathcal{Z}$.

We also define the shortcut

```math
p(y, \ell, u) = \ell^\top y^+ - u^\top y^-
```

which leaves us with

```math
\mathcal{L}(x, y, z) = (c - A^\top y - z)^\top x + p(y; \ell_c, u_c) + p(z; \ell_v, u_v)
```

From there, we deduce the dual problem:

```math
\max_{y, z} \quad p(y; \ell_c, u_c) + p(z; \ell_v, u_v) \quad \text{s.t.} \quad \begin{cases}
0 = c - A^\top y - z \\
y \in \mathcal{Y} \\
z \in \mathcal{Z}
\end{cases}
```

The primal-dual gap (one of our stopping criteria) thus writes as

```math
g = c^\top x - \left(p(y; \ell_c, u_c) + p(z; \ell_v, u_v)\right)
```

## Preconditioning

The original problem $P$ and preconditioned problem $\tilde{P}$ are linked by:

- Constraint matrix $\tilde{A} = D_1 A D_2$ so $A = D_1^{-1} \tilde{A} D_2^{-1}$
- Transposed constraint matrix $\tilde{A}^\top = D_2 A^\top D_1$ so $A^\top = D_2^{-1} \tilde{A}^\top D_1^{-1}$
- Primal variable $\tilde{x} = D_2^{-1} x$ so $x = D_2 \tilde{x}$
- Dual variable for constraints $\tilde{y} = D_1^{-1} y$ so $y = D_1 \tilde{y}$, but $\tilde{\mathcal{Y}} = \mathcal{Y}$
- Dual variable for bounds $\tilde{z} = D_2 z$ so $z = D_2^{-1} \tilde{z}$, but $\tilde{\mathcal{Z}} = \mathcal{Z}$
- Cost $\tilde{c} = D_2 c$ so $c = D_2^{-1} \tilde{c}$
- Bounds $(\tilde{\ell}_v, \tilde{u}_v) = D_2^{-1} (\ell_v, u_v)$ so $(\ell_v, u_v) = D_2 (\tilde{\ell}_v, \tilde{u}_v)$
- Constraints $(\tilde{\ell}_c, \tilde{u}_c) = D_1 (\ell_c, u_c)$ so $(\ell_c, u_c) = D_1^{-1} (\tilde{\ell}_c, \tilde{u}_c)$

Then we have the following terms in the KKT errors:

```math
\begin{align*}
c - A^\top y - z
& = D_2^{-1} \tilde{c} - D_2^{-1} \tilde{A}^\top D_1^{-1} D_1 \tilde{y} - D_2^{-1} \tilde{z} \\
& = D_2^{-1}(\tilde{c} - \tilde{A}^\top \tilde{y} - \tilde{z})
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
z - \mathrm{proj}_{\mathcal{Z}}(z) = D_2^{-1} \tilde{z} - \mathrm{proj}_{\tilde{\mathcal{Z}}}(D_2^{-1} \tilde{z}) = D_2^{-1} (\tilde{z} - \mathrm{proj}_{\tilde{\mathcal{Z}}}(\tilde{z}))
```

```math
c^\top x = (D_2^{-1} \tilde{c})^\top (D_2 \tilde{x}) = \tilde{c}^\top D_2^{-1} D_2 \tilde{x} = \tilde{c}^\top \tilde{x}
```

```math
\begin{align*}
p(y; \ell_c, u_c)
& = \ell_c^\top y^+ - u_c^\top y^- \\
& = (D_1^{-1} \tilde{\ell}_c)^\top (D_1 \tilde{y})^+ - (D_1^{-1} \tilde{u}_c)^\top (D_1 \tilde{y})^- \\
& = \tilde{\ell}_c^\top D_1^{-1} D_1 \tilde{y}^+ - \tilde{u}_c^\top D_1^{-1} D_1 \tilde{y}^- \\
& = \tilde{\ell}_c^\top \tilde{y}^+ - \tilde{u}_c \tilde{y}^-
\end{align*}
```

```math
\begin{align*}
p(z; \ell_v, u_v)
& = \ell_v^\top z^+ - u_v^\top z^- \\
& = (D_2 \tilde{\ell}_v)^\top (D_2^{-1} \tilde{z})^+ - (D_2 \tilde{u}_v)^\top (D_2^{-1} \tilde{z})^- \\
& = \tilde{\ell}_v^\top D_2 D_2^{-1} \tilde{z}^+ - \tilde{u}_v^\top D_2 D_2^{-1} \tilde{z}^- \\
& = \tilde{\ell}_v^\top \tilde{z}^+ - \tilde{u}_v^\top \tilde{z}^-
\end{align*}
```

We make use of a few key observations:

- Projection on $\mathcal{Z}$ commutes with scaling
- Projection on an interval commutes with scaling if scaling is also applied to the interval in question
