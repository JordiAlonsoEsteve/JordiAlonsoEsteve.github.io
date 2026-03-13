---
layout: post
title: "LASSO: Subgradient method, projected gradient descent, proximal gradient descent and coordinate descent."
tag: [Optimization, ML]
---

<link href="/css/syntax.css" rel="stylesheet">

We will cover, superficially, the main optimization approaches for a l1 penalized regression. Jupyter notebook with the implementation can be found [here](/_posts/Optimization/2026-03-13-LASSO-algos.md).

## The Least Absolute Shrinkage and Selection Operator (LASSO) problem

Let $\mathbf{X} \in \mathbb{R}^{N \times p}$, $\mathbf{y} \in \mathbb{R}^{N}$, being $p$ the number of "features" and $N$ the number of "observations". The LASSO problem is:

$$\begin{aligned}
\min_{\beta \in \mathbb{R}^p} \quad & \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\beta\|_2^2 \\
\text{subject to} \quad & \|\beta\|_1 \leq t
\end{aligned}$$

Which, as long as the constrain is active, we have a one-to-one correspondence with a Lagrangian formulation in terms of $$\lambda$$:

$$\min_{\beta \in \mathbb{R}^p} \left\{ \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\beta\|_2^2 + \lambda \|\beta\|_1 \right\}$$

This is probably one of the easiest interesting problems, stemming from the fact that the absolute value $$ \vert x \vert$$ is not a differentiable function at $$x = 0$$. This is rather obvious since

$$\lim_{h \to 0^+} \frac{|0 + h| - |0|}{h} \neq  \lim_{h \to 0^-} \frac{|0 + h| - |0|}{h}.$$

## Subgradient descent
A workaround (which easily holds for convex functions) is the subgradient:

A vector $g \in \mathbb{R}^n$ is a subgradient of $f:\mathbb{R}^n \rightarrow \mathbb{R}$ at $x \in \text{dom}\, f$ if 
$$f(z) \geq f(x) + g^{\top}(z - x)$$
$\forall z \in \text{dom} \,f.$ The set of all subgradients at $x$ is refered to as subdifferential ($\partial f(x)$).

For $\vert x\vert$ is actually very simple:

$$\partial \vert x\vert = 
\begin{cases} 
\{1\} & \text{if } x > 0 \\ 
\{-1\} & \text{if } x < 0 \\ 
[-1, 1] & \text{if } x = 0 
\end{cases}$$

Now, we have that:

$$ 0 \in \partial f(x) \implies f(y) \geq f(x) \, \forall y \in \text{dom} \, f.$$

And hence that is a condition for optimality.

This allows to proceed with a subgradient method approach, akin to gradient descent! However, the convergence rate of this algorithm is $\mathcal{O}(\frac{1}{\sqrt{k} })$ for $k$ being the number of iterations. This is remarcably worse than gradient descent, which is an interesting result in itself.

The algorithm is very simple:

looking at the lasso problem, taking the derivative:

$$\frac{\partial \big (\frac{1}{2n} \|\mathbf{y} - \mathbf{X}\beta\|_2^2 + \lambda \|\beta\|_1 \big )}{\partial \beta} = -\mathbf{X}^{\top}\mathbf{y} + \mathbf{X}^{\top}\mathbf{X}\beta +  \lambda \partial \|\beta\|_1.$$

And we would be kind of done. However, the step size is a tricky issue in this case. In fact, this is probably one of the most interesting bits, although we will not dive in the theory here (will be covered in some other notes).

For a convex and L-smooth function (where the gradient is Lipschitz continuous with constant L), gradient gescent is guaranteed to converge to the global optimum using a fixed step size $\eta$, provided that $0<\eta\leq\frac{1}{L}$ at a rate $\mathcal{O}(\frac{1}{k})$. However, this is **not** the case for the subgradient method. The guarantee of convergence to the optimum in the non-differentiable case requires diminishing step sizes, an easy option that works is, for iteration $k$:

$$
\eta_k = \eta_0 / \sqrt{k}.
$$

In the LASSO case, this has a clear intuition. For a fixed step we will never reach actual 0s because the subgradient of the absolute value​ remains constant in magnitude regardless of how close a coefficient is to the origin (which is very different from an l2 norm, for example). Instead of settling, the iterates will simply "bounce" across the axis in a perpetual oscillation of size proportional to $\eta \lambda$.

For example, 50 samples 100 dimensions, and some small noise, starting at $\eta = 0.1$. Only the first 10 dimensions are non zero but the subgradient approach gets no actual zeros!

### Projected gradient descent

We are back at the constrained optimization problem formulation. The idea is to keep $\beta$ inside of the set described by the constraint $\|\beta\|_1 \leq t$. This is, while applying the gradient step might get the parameters out of the set, we will push them back intto it. In particular, we will find the element in the set that is closest to our current value in terms of euclidean distance.

The projection operator $P_C$ is defined as:

$$P_C(x) := \arg\min_{z \in C} \frac{1}{2} \|z - x\|_2^2,$$

where $$C = \{\beta :\|\beta\|_1 \leq t \}$$.

The actual solution to $P_C(x)$ is not trivial (see [2])
<p align="center">
  <img src="/assets/images/LASSO/l1_projection.png" alt="l1 projection"/>
</p>

Somewhat suprisingly, the convergence rate of projected gradient descent is the same as gradient descent: $\mathcal{O}(\frac{1}{k})$

In this case we get ~70 zeroes. 

### Proximal gradient descent

The second most common approach (used in ISTA, FISTA...) is using a proximal operator:

$$\text{Prox}_h(z) :=  \arg\min_{\theta \in \mathbb{R}^p} \frac{1}{2} \|z - \theta\|_2^2 + h(\theta)$$


If $h(\theta)$ becomes the indicator function for a set $C$ we recover projected gradient descent, so this is indeed a generalization. The interesting bit with this is that the solution for that operator when $$h(\theta) = \|\theta\|_1$$ is a piece-wise soft-treshold operation. This is extremely fast.

Arriving at the soft-tresholding function is kind of fun;

$$
\begin{align*}

\frac{1}{2} \|z - \theta\|_2^2 + h(\theta) &= \\
\frac{1}{2} \sum_i^N \left [ (z_i - \theta_i)^2 + 2\lambda |\theta_i|\right ] & = \\
\frac{1}{2} \sum_i^N \left [ z_i^2 - 2 z_i \theta_i + \theta_i^2 + 2\lambda |\theta_i|\right ] \propto \\
\frac{1}{2} \sum_i^N \left [ - 2 z_i \theta_i + \theta_i^2 + 2\lambda |\theta_i|\right ]. \\
\end{align*}
$$

Finding the minimum:

$$
\sum_i^N \left [ - z_i + \theta_i + \lambda \partial |\theta_i|\right ]  = 0, \\
$$

using the subgradient definition, three options appear:

* **Case 1:** If $\theta_i > 0 \implies -z_i + \theta_i + \lambda = 0 \implies \theta_i = z_i - \lambda$ (Valid only if $z_i > \lambda$)
* **Case 2:** If $\theta_i < 0 \implies -z_i + \theta_i - \lambda = 0 \implies \theta_i = z_i + \lambda$ (Valid only if $z_i < -\lambda$)
* **Case 3:** If $\theta_i = 0 \implies z_i = 0$

But, clearly, $z_i < 0 \implies \theta_i < 0$ and $z_i > 0 \implies \theta_i > 0$. Otherwise we won't be minimizing! Hence, we have an implication that defines the value of $\theta_i$ from the known value of $z_i$. In particular, this boils down to:

$$ \text{Prox}_{\lambda \|\cdot\|_1}(z_i) = S_\lambda(z_i) = \text{sign}(z_i) \cdot (|z_i| - \lambda)_+ $$

This yields ISTA, with convergence rate $\mathcal{O}(\frac{1}{k})$.

Again, we get ~70 zeroes.

### Coordinate descent

This is the most common, implemented in glmnet, used in sklearn and so on. The convex, separable objective function for LASSO can be optimized one parameter at a time:

$$
\beta_k^{t+1}
=
\arg\min_{\beta_k}
f(\beta_1^{t}, \beta_2^{t}, \dots, \beta_{k-1}^{t}, \beta_k, \dots, \beta_p^{t}).
$$

This has a closed form solution using an argument very close to the ISTA one using the Soft-tresholding. Consider the LASSO problem

$$
\min_{\beta}
\frac{1}{N}\sum_{i=1}^{N}
\left(
y_i - \sum_{k=1}^{p} x_{ik}\beta_k
\right)^2
+
\sum_{k=1}^{p} \lambda |\beta_k|,
$$

define:

$$
r_i^{(j)} :=
y_i - \sum_{k \ne j} x_{ik}\beta_k.
$$

Then the problem over each coordinate becomes:

$$
\min_{\beta_j}
\frac{1}{N}\sum_{i=1}^{N}
(r_i^{(j)} - x_{ij}\beta_j)^2
+
\lambda |\beta_j|
$$


Now, expanding and taking the derivative;

$$
\frac{1}{N}\sum_{i=1}^{N}
\left[
r_i^{(j)2}
-2x_{ij}r_i^{(j)}\beta_j
+
x_{ij}^2\beta_j^2
\right]
+
\lambda |\beta_j|,
$$

$$
-\frac{1}{N}\sum_{i=1}^{N} x_{ij} r_i^{(j)}
+
\frac{1}{N}\sum_{i=1}^{N} x_{ij}^2 \beta_j
+
\lambda \, \partial |\beta_j|
=0
$$

Let

$$
a = \frac{1}{N}\sum_{i=1}^{N} x_{ij}^2, \quad
b = \frac{1}{N}\sum_{i=1}^{N} x_{ij} r_i^{(j)}
$$

Then

$$
a \beta_j + \lambda \partial |\beta_j| = b.
$$

Case 1: $\beta_j > 0$ 

$$
\beta_j a + \lambda = b \Rightarrow \beta_j = \frac{b - \lambda}{a}
$$

If $b > 0 \Rightarrow \beta_j > 0$, otherwise we have a contradiction, since $a$ and $\lambda$ are positive scalars.

Case 2: $\beta_j < 0$

$$
\beta_j a - \lambda = b
\Rightarrow \beta_j = \frac{b + \lambda}{a}
$$

Case 3: $\beta_j = 0$


which leads to the **soft-thresholding update**:

$$
\hat{\beta}_j =
\frac{
S_\lambda
\left(
\frac{1}{N}\sum_{i=1}^{N} r_i^{(j)} x_{ij}
\right)
}{
\frac{1}{N}\sum_{i=1}^{N} x_{ij}^2
}
$$

This approach is nice because it finds the optimal solution coordinate wise (conditioned on the other parameters). While the implementation we use is naive and consists on recursively applying the above equation on a nested loop of iterations and parameters, SOTA implementations exploit the sparsity in cool ways to make this faster.


### Putting all together

We can check the performance of the algorithms evaluating the fit to test data and the distance from the actual parameters, by the optimization step.

Note that projected and proximal gradient descent were implemented with backtracking using Armijo's rule.


Coordinate descent is, in this case, much faster, reaching the tolerance level of $1 \times 10^{-16}$ before the 500 iterations limit. Although the comparison is not fair because the iterations are not really comparable. Regardless, it took less time than the others. 

<p align="center">
  <img src="/assets/images/LASSO/performance.png" alt="l1 projection"/>
</p>

Note that projected gradient has the perfect "penalty" parameter $t = \|\beta_i\|_1$, $\lambda$ has been extracted by trial and error.

IIn any case, the limitations of my NumPy implementations likely dominate many of the observed performance differences. The only robust conclusion is that the subgradient method is clearly suboptimal for the vanilla LASSO problem and offers little practical justification here. In particular, one should be careful not to ignore the non-differentiability of the l1 norm For example, performing backpropagation + "gradient descent" through an absolute value function (PyTorch will allow without complaint) perhaps deserves some thought!



## References
1. **Hastie, T., Tibshirani, R., & Wainwright, M. (2015).**  

   *[Statistical learning with Sparsity](https://hastie.su.domains/StatLearnSparsity/)*

2. **Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and Chandra, Tushar (2008)**

    *[Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions](https://ai.stanford.edu/~jduchi/projects/jd_ss_ys_l1.pdf)*

3. **Ryan Tibshirani**

    *[Subgradient Method lecture notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/sg-method.pdf)*
    
    *[Gradient descent lecture notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec6.pdf)*
    

