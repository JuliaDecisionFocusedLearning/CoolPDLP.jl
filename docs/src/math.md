# Math

> See the references in the README for details, but beware that we adopt slightly different notational conventions.

## Primal and dual problems

PDLP solves Linear Programs (LPs) formulated as follows:

```math
\min_x \quad c^\top x + c_0 \quad \text{s.t.} \quad \begin{cases}
\ell_c \leq A x \leq u_c \\
\ell_v \leq x \leq u_v
\end{cases}
```

We associate non-negative multipliers $y_\ell, y_u, z_\ell, z_u \geq 0$ with all four inequality constraints, leading to the following Lagrangian:

```math
\begin{align*}
\mathcal{L}(x, y_\ell, y_u, z_\ell, z_u)
& = c^\top x + c_0 + y_\ell^\top (\ell_c - A x) + y_u^\top (A x - u_c) + z_\ell^\top (\ell_v - x) + z_u^\top (x - u_v) \\
& = (c - A^\top y_\ell + A^\top y_u - z_\ell + z_u)^\top x + (y_\ell^\top \ell_c - y_u^\top u_c) + (z_\ell^\top \ell_v - z_u^\top u_v) + c_0
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
\mathcal{L}(x, y, z) = (c - A^\top y - z)^\top x + p(y; \ell_c, u_c) + p(z; \ell_v, u_v) + c_0
```

From there, we deduce the dual problem:

```math
\max_{y, z} \quad p(y; \ell_c, u_c) + p(z; \ell_v, u_v) + c_0 \quad \text{s.t.} \quad \begin{cases}
0 = c - A^\top y - z \\
y \in \mathcal{Y} \\
z \in \mathcal{Z}
\end{cases}
```

The primal-dual gap (one of our stopping criteria) thus writes as

```math
g = c^\top x - \left(p(y; \ell_c, u_c) + p(z; \ell_v, u_v)\right)
```

The objective constant $c_0$ shifts the primal and dual objectives alike, so it cancels out in the gap.
It does not cancel out in the scale $1 + |c^\top x + c_0| + |p(y; \ell_c, u_c) + p(z; \ell_v, u_v) + c_0|$ that the gap is compared to.

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

### Pock-Chambolle scaling

Following Lemma 2 of:

> Pock, Thomas, and Antonin Chambolle. "Diagonal preconditioning for first order primal-dual algorithms in convex optimization." *2011 International Conference on Computer Vision*. IEEE, 2011. [doi:10.1109/ICCV.2011.6126441](https://doi.org/10.1109/ICCV.2011.6126441)

for matrix $K$ of size $m \times n$ and $\alpha \in [0, 2]$, the diagonal factors are

```math
\tau_j = \left(\sum_i |K_{ij}|^{\alpha}\right)^{-1} \quad \text{(variable side, column $j$)}
```

```math
\sigma_i = \left(\sum_j |K_{ij}|^{2-\alpha}\right)^{-1} \quad \text{(constraint side, row $i$)}
```

so that $\|\Sigma^{1/2} K T^{1/2}\| \leq 1$ with $T = \mathrm{diag}(\tau_j), \Sigma = \mathrm{diag}(\sigma_i)$. CoolPDLP absorbs the $1/2$ powers into the rescaling factors: $D_2[j] = \tau_j^{1/2}$ and $D_1[i] = \sigma_i^{1/2}$. The exponent parameter $\alpha$ is exposed as `chambolle_pock_alpha`. See also [`CoolPDLP.chambolle_pock_preconditioner`](@ref).

Note that CoolPDLP uses $2-\alpha$ for the row and $\alpha$ for the column,
to be aligned with the [PDLP paper](https://dl.acm.org/doi/abs/10.5555/3540261.3541809). This is the opposite of the
convention used in the original Pock-Chambolle paper referenced above.

### Ruiz equilibration

The Ruiz iteration alternately rescales each row and each column of $A$ by the square root of its $\ell_\infty$ norm. After enough iterations the rescaled matrix has all row and column $\infty$-norms close to 1. See also [`CoolPDLP.ruiz_preconditioner`](@ref).

### Composition

The diagonal preconditioner chains the following stages in this order, each acting on the matrix scaled by the previous ones:

- `curtis_reid_iter` rounds of Curtis–Reid least-squares scaling, which minimizes $\sum_{ij} (\log|a_{ij}| + r_i + s_j)^2$ over the nonzeros by block coordinate descent (not the faster conjugate-gradient method of Curtis and Reid, so it may stop short of the minimum on badly scaled matrices);
- `geometric_mean_iter` passes of geometric-mean scaling, dividing each row, then each column, by $\sqrt{\min |a| \cdot \max |a|}$;
- `ruiz_iter` Ruiz passes and the Pock–Chambolle step above.

`Algorithm` turns the first two off by default, while `PDLP()` runs 20 Curtis–Reid rounds. Their factors are clamped to $[10^{-20}, 10^{20}]$.

With `bound_objective_rescaling = true`, the preconditioned bounds are then multiplied by $\beta = 1 / (1 + \|b\|_2)$ and the objective by $\gamma = 1 / (1 + \|c\|_2)$, per instance, as in cuPDLPx and HPR-LP.
A solution $(x, y)$ of the original problem becomes $(\beta x, \gamma y)$, PDLP starts from a primal weight of one, and errors and solutions are still reported for the original problem.

The factors of the stages multiply: if stage $k$ produces $D_1^{(k)}, D_2^{(k)}$, the final scaling is

```math
D_1 = \cdots D_1^{(2)} D_1^{(1)} \qquad D_2 = D_2^{(1)} D_2^{(2)} \cdots
```

## Termination

With $\epsilon_a$ = `termination_abstol` and $\epsilon_r$ = `termination_reltol`, an instance is optimal once

```math
\|Ax - \mathrm{proj}_{[\ell_c,u_c]}(Ax)\| \leq \epsilon_a + \epsilon_r \|b\|, \qquad
\|c - A^\top y - z\| \leq \epsilon_a + \epsilon_r \|c\|, \qquad
|p - d| \leq \epsilon_a + \epsilon_r (|p| + |d|),
```

where $b$ keeps the largest finite absolute bound of each constraint, $p$ and $d$ are the primal and dual objectives, and the norm is `termination_norm` (`L2Norm()` or `LInfNorm()`, `"L2"` or `"Linf"` through MOI).
This is the criterion of cuPDLP.jl and cuOpt.
The default $\epsilon_a = \epsilon_r$ gives back the purely relative criterion $\text{error} / (1 + \text{norm}) \leq \epsilon_r$ of PDLP.
Every quantity is that of the original problem, whatever the preconditioning.
Restarts always compare errors in the $\ell_2$ norm.
## Infeasibility detection

When a problem is infeasible, the iterates diverge, and their differences converge to a direction whose primal or dual part certifies it ([Applegate et al.](https://arxiv.org/abs/2102.04592)).
At every termination check, two candidate directions (for PDLP, the last step and the displacement since the last restart; for PDHG, the last step and the iterate itself) are first projected onto the sign conditions of a ray, then tested on the original problem:

- a dual ray $y$, with reduced cost $r = -A^\top y$, certifies primal infeasibility when $r$ only pairs with finite variable bounds and $p(y; \ell_c, u_c) + p(r; \ell_v, u_v) > 0$;
- a primal ray $d$ certifies dual infeasibility when $d$ and $Ad$ lie in the recession cones of $[\ell_v, u_v]$ and $[\ell_c, u_c]$ and $c^\top d < 0$.

The error of a ray is the norm of its violation of these cone conditions divided by its objective, which does not depend on its length, and a ray qualifies only if its objective is at least `infeasibility_tol` times the sum of the absolute values of its terms, which rules out rounding noise.
An instance is `MOI.INFEASIBLE` (resp. `MOI.DUAL_INFEASIBLE`) once that error is at most `infeasibility_tol` (default $10^{-8}$), optimality taking precedence, then primal infeasibility.
The certificates are returned in `stats.ray`, in the original space: `stats.ray.y` for primal infeasibility, `stats.ray.x` for dual infeasibility.

Each check costs two more KKT passes (one product by $A$ and $A^\top$ per candidate), counted in `stats.ray_passes` rather than in `stats.kkt_passes`, so they do not count towards `max_kkt_passes`.
With `infeasibility_tol = nothing`, the detection is removed at the type level: no buffers, no extra passes, and nothing to trace under Reactant.
Crossed bounds ($\ell > u$) are reported as infeasible before the solve, without a certificate.

