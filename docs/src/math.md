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

The diagonal preconditioner chains, in this order, `curtis_reid_iter` rounds of Curtis–Reid least-squares scaling (block coordinate descent, not the conjugate-gradient method of Curtis and Reid, on $\sum_{ij} (\log|a_{ij}| + r_i + s_j)^2$ over the nonzeros), `geometric_mean_iter` passes of geometric-mean scaling (each row, then each column, divided by $\sqrt{\min |a| \cdot \max |a|}$), `ruiz_iter` Ruiz passes and a Pock–Chambolle step (see above).
Each stage acts on the matrix scaled by the previous ones. `Algorithm` turns the first two off by default, while `PDLP()`, `HPDLP()` and `HPRLP()` start with 20 Curtis–Reid rounds: `PDLP(; curtis_reid_iter = 0)` gives back plain Ruiz and Pock–Chambolle scaling. The factors of both are clamped to $[10^{-20}, 10^{20}]$.

With `bound_objective_rescaling = true`, the preconditioned bounds are then multiplied by $\beta = 1 / (1 + \|b\|_2)$ and the objective by $\gamma = 1 / (1 + \|c\|_2)$, per instance, as in cuPDLPx and HPR-LP.
A solution $(x, y)$ of the original problem becomes $(\beta x, \gamma y)$, and the objective is multiplied by $\beta \gamma$.
PDLP and HPDLP then start from a primal weight of one.
`bound_norm_sum_squares = true` takes the cuPDLPx definition of $\|b\|_2^2$, which sums $\ell^2 + u^2$ over the finite bounds.

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

where $b$ keeps the largest finite absolute bound of each constraint, $p$ and $d$ are the primal and dual objectives, and the norm is `termination_norm` (`:L2` or `:Linf`).
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

## Halpern methods

`HPDLP` and `HPRLP` share one outer loop. Write $T$ for an operator that is firmly nonexpansive in a metric $P$. Each step is reflected and anchored (Halpern):

```math
z^{k+1} = \frac{k+1}{k+2} \left( \gamma \, (2 T(z^k) - z^k) + (1 - \gamma) \, z^k \right) + \frac{1}{k+2} \, z^0
```

where $k$ counts steps since the last restart and $z^0$ is the anchor. $T(z^k)$ is the reported point.
Every `check_every` steps, the fixed-point residual $E = \|z - T(z)\|_P$ decides restarts. An instance restarts when either:

- $E \leq \beta_{\text{suff}} E_0$, or
- $E \leq \beta_{\text{nec}} E_0$ and $E$ grew since the last check, or
- the inner loop is long: $k \geq \beta_{\text{art}}$ times the total number of steps.

Here $E_0$ is the residual at the first step after the last restart. A restart moves both $z$ and $z^0$ to $T(z)$. In a batch, every instance restarts on its own.

### HPDLP: reflected Halpern PDHG (r²HPDHG)

$T$ is the PDHG step of `pdhg_update!`, with $\tau = \eta/\omega$ and $\sigma = \eta\omega$. With the sign convention of $y$ used here, its metric is $P = \begin{bmatrix} I/\tau & A^\top \\ A & I/\sigma \end{bmatrix}$, and the residual is measured as

```math
\eta \|\Delta\|_P^2 = \omega \|\Delta x\|^2 + \|\Delta y\|^2/\omega + 2\eta \langle \Delta x, A^\top \Delta y\rangle.
```

At a restart, PID control in log space sets the primal weight. The error is $e = \log(\|\Delta y\| / (\omega \|\Delta x\|))$, measured between $T(z)$ and the previous anchor. The update is $\omega \leftarrow \omega \exp(k_P e + k_I s + k_D (e - e_{\text{prev}}))$, with $s \leftarrow \zeta s + e$.

| | cuPDLPx | `HPDLP()` |
|---|---|---|
| step size | $0.998/\|A\|_2$ | same |
| restart decays (suff., nec., art.) | 0.2, 0.5, 0.36 | same |
| check frequency | 200 | same |
| PID $(k_P, k_I, k_D, \zeta)$ | (0.99, 0.01, 0, 0.3) | same |
| preconditioning | Ruiz 10 + Pock-Chambolle | **Curtis-Reid 20** + Ruiz 10 + Pock-Chambolle |
| bound/objective rescaling | on | **off** (on costs about 15 % more passes on Netlib) |
| ray candidates | $T(z) - z^0$, $T(z)$, $T(z) - z$ | first two only |

### HPRLP: HPR-LP

With $T_1 = \lambda I - AA^\top$ and $\lambda \geq \lambda_{\max}(AA^\top)$, an HPR-LP step is a PDHG step. Its primal step is $\sigma$ and its dual step is $1/(\lambda\sigma)$, so the code uses $\eta = 1/\sqrt\lambda$ and $\omega = \eta/\sigma$.
The metric $M$ of the splitting gives $\|\Delta\|_M^2 = \|\Delta x\|^2/\sigma + \sigma\lambda \|\Delta y\|^2 + 2\langle A\Delta x, \Delta y\rangle$, which is the PDHG metric above divided by $\eta$.
The implementation follows the reference code (v0.1.7) rather than the paper where the two differ. On Netlib, its restart iterations and values of $\sigma$ match the reference code's exactly.

| | paper | reference code | `HPRLP()` |
|---|---|---|---|
| $\lambda$ | power method | 1.01 × power method, grown on the fly when $\|\Delta\|_M^2 < 0$ | as the code (never for a frozen instance) |
| $\sigma_0$ | 1 | $\|b\|/\|c\|$ of the scaled problem | as the code |
| $\sigma$ update | $\|\Delta x\|/(\|\Delta y\|\sqrt\lambda)$ | blended in log space with the $\sigma$ of the best residual, plus a correction near $10^{-9}$ | as the code |
| first check | rule-based | always restarts | as the code |
| preconditioning | Ruiz 10 + Pock-Chambolle + $b, c$ scaling | Curtis-Reid 20 first | as the code |
| termination | every 150 steps | also on the logging schedule | every 150 steps |
| dual residual | | uses $\bar z$ of the splitting | as the code |
| infeasibility | none | none | ray certificates |

## Quadratic programs

With a symmetric positive semidefinite objective matrix $Q$, the primal problem becomes

```math
\min_x \tfrac{1}{2} x^\top Q x + c^\top x + c_0 \quad \text{s.t.} \quad \ell_c \leq Ax \leq u_c, \quad \ell_v \leq x \leq u_v
```

and, following [PDQP](https://arxiv.org/abs/2311.07710) and [HPR-QP](https://github.com/PolyU-IOR/HPR-QP), the KKT errors change as follows:

- The reduced costs are $z = \mathrm{proj}_{\mathcal{Z}}(c + Qx - A^\top y)$, and the dual residual is $c + Qx - A^\top y - z$.
- The primal objective is $\tfrac{1}{2} x^\top Q x + c^\top x + c_0$ and the dual objective is $p(y; \ell_c, u_c) + p(z; \ell_v, u_v) - \tfrac{1}{2} x^\top Q x + c_0$, so the gap is $x^\top Q x + c^\top x - p(y; \ell_c, u_c) - p(z; \ell_v, u_v)$.

Preconditioning scales $\tilde{Q} = D_2 Q D_2$, so that $\tfrac{1}{2} x^\top Q x = \tfrac{1}{2} \tilde{x}^\top \tilde{Q} \tilde{x}$ and $Qx = D_2^{-1} \tilde{Q} \tilde{x}$, which is unscaled like the rest of the dual residual.
The column norms of the Ruiz and Pock-Chambolle preconditioners are those of $[Q; A]$.

PDHG, PDLP and HPDLP linearize the objective in the primal step, $x^+ = \mathrm{proj}_{[\ell_v, u_v]}(x - \tau (c + Qx - A^\top y))$.
This is a proximal point step in the metric $\begin{bmatrix} I/\tau - Q & A^\top \\ A & I/\sigma \end{bmatrix}$ (with the sign convention of $y$ used here; the sign of the off-diagonal blocks does not change the condition below), which is positive semidefinite as soon as $1/\tau - \|Q\| \geq \sigma \|A\|^2$.
With $\tau = \eta / \omega$ and $\sigma = \eta \omega$, the step size is thus

```math
\eta = \frac{2}{\|Q\|/\omega + \sqrt{\|Q\|^2/\omega^2 + 4 \|A\|^2}}
```

times `invnorm_scaling`, recomputed whenever the primal weight $\omega$ changes.
HPDLP measures its fixed-point residual in the same metric, which adds $-\eta \langle \Delta x, Q \Delta x \rangle$ to the one of linear programs.
A linear program keeps a `ZeroMatrix` as $Q$, on which all of the above dispatches back to the linear code.

An algorithm that does not support quadratic objectives (see [`CoolPDLP.supports_quadratic`](@ref)) throws an `ArgumentError` on a quadratic program.

## Step size

The fixed step size is $\eta = s / \|\tilde{A}\|_2$, where $s$ is `invnorm_scaling` and PDHG needs $\eta \|\tilde{A}\|_2 < 1$ to converge.
The spectral norm is an iterative estimate $\sigma$, accepted once $\tilde{A}$ provably has a singular value within $\tau \sigma$ of it, with $\tau$ = `spectral_norm_reltol`.
When $\sigma$ approximates the top of the spectrum, it lies below $\|\tilde{A}\|_2$ by at most about $\tau$ relative, a margin absorbed by $s \leq 1 - \tau$, which holds comfortably for the default $s = 0.9$.

If the estimate does not converge within `spectral_norm_maxiter` iterations, a warning is shown and the estimate falls back to the smaller of $1.01 (\sigma + \text{residual})$ and the rigorous bound $\sqrt{\|\tilde{A}\|_1 \|\tilde{A}\|_\infty}$.
