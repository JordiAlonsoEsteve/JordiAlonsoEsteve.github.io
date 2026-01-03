---
layout: post
title: Inequality constrained optimization (SVM "the technical problem").
tag: [ML, Optimization]
---

<link href="/css/syntax.css" rel="stylesheet" >
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# From Optimal Separating Hyperplane to Support Vector Machines
Previously we already explained the basics of [Optimal Separating Hyperplanes](/2024/01/16/Understanding-SVMs_1.html). There we considered the problem of optimally separating hyperplpanes also with expansions of the input space, which allowed perfect separability of the classes. That extension was already in Support Vector Machines (SVMs) territory, but we did not consider the soft margin. Furthermore, there we used an off-the-shelf library for the QP problem. Here we slightly generalize the framework to that of the SVMs. The focus here, however, is on the optimization part of the problem, on the "technical" side. We will dive with quite a lot of detail in inequality constraint optimization theory for convex problems, being the SVM problem just an illustration (a rather simple one) of the applicability of those algorithms.

In essence SVM is a optimal separating hyperplane with two additions; First, the mapping of the input space to a high dimensional featuere space (e.g. polynomial expansion of degree $$d$$). Second, soft margin classification, which allows non-perfect separability of the classes. 

After solving for the primal and plugging in the solution, we have the following dual problem for the optimal separating hyperplane:

$$
\begin{align*}
\text{maximize}_{\alpha} \quad & 
\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
\text{subject to} \quad & 
\sum_{i=1}^n \alpha_i y_i = 0, \\
& \alpha_i \geq 0, \quad i = 1, \dots, n.
\end{align*}
$$

Generalizing this to the soft margin hyperplane is very simple, introducing a complexity parameter $$C$$:

$$ 
\begin{align*}
\text{maximize}_{\alpha} \quad & 
\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
\text{subject to} \quad & 
\sum_{i=1}^n \alpha_i y_i = 0, \\
& 0 \le \alpha_i \le C, \quad i = 1, \dots, n.
\end{align*}
$$

Intuitively, $$C$$ is an upper bound on the lagrange multiplier associated to an observation. It limits the importace of the constraint of proper classification of the points. The higher $$C$$ the more importance it has to missclasify, with a very big $$C$$ we recover the hard-margin classifyier because the "natural" lagrange multipliers will be below $$C$$.

The combination of basis expansion with maximum margin separation is extremely interesting from the statistical learning perspective. The conceptual motivation for SVMs is a very rich topic. Rather than selecting a hypothesis by directly controlling model complexity through the number of parameters or network architecture, SVMs introduce an inductive bias based on margin maximization. The inductive step is taken by restricting attention to classifiers that achieve zero empirical risk on the training data (in the separable case), and among these, choosing the one that minimizes an upper bound on the probability of misclassification on unseen data. This shift, from fitting capacity to maximum margin, marks a fundamental departure from more traditional forms of inductive bias. 

We will dive into the "conceptual" problem of SVMs in a future post.


For now, let's dive into inequality constrained convex optimization. Note that an explanation of Newton's method for equality constrained optimization is available [here](/2025/12/25/SteepestDescent.html). We build from there.

# Inequality constrained optimization
Often when looking into how SVM works, at some point a QP solver is invoked. We will code that explicltly. Here we illustrate two interior point methods, ADMM and an ad-hoc algorithm for SVMs. In the first two points we loosely follow chapter 11 from [1].

Code available here . TODO The idea is to use only numpy.

## Log-Barrier method
It is not immediately clear how one can go to solve an unequality constrained minimization problem. Equality constrained minimization seems more easier, we just have to make updates orthogonal to the constraint matrix. Perhaps, the first thing that one can try is to add a penalty that goes to $$\infty$$ assuming we are minimizing (wlog).

The Log Barrier method is a rigurous approach in that direction.

Define the barrier functional:

$$
\phi(x) = - \sum_{i=1}^m \log{(-f_i(x))}.
$$

This function clearly goes to $$\infty$$ when $$x \rightarrow 0$$, and it is only defined for $$f_i(x) < 0$$. Hence, this is a perfect penalty to add to our objective function, transforming the inequality constrained optimization problem into an equality constrained optimization one. However, to not change the objective we want to optimize, we need to make the penalty very small when the parameters are feasible. To do that a constant "small" $$t > 0$$ is added.

The gradient and Hessian of $$\phi(x)$$ are very important:

$$
\begin{align*}
\nabla \phi(x) &= -\sum_{i=1}^m \frac{1}{f_i(x)} \, \nabla f_i(x) \\

\nabla^2 \phi(x) &= \sum_{i=1}^m \frac{1}{f_i(x)^2} \, \nabla f_i(x) \, \nabla f_i(x)^\top
+ \sum_{i=1}^m \frac{1}{f_i(x)} \, \nabla^2 f_i(x)
\end{align*}
$$

For the SVM case we have two inequality constraints $$f^i_1(\alpha) = -\alpha_i \leq 0$$ ($$\alpha$$ must be positive) and $$f^i_2(\alpha) = \alpha_i - C \leq 0$$ ($$\alpha$$ must be below $$C$$). In this case the barrier functional is (note that we are maximizing so the first sign is changed to look for $$-\infty$$):

$$
\phi(\alpha) = \sum_{i=1}^m \big( \log{(-f^i_1(x))} + \log{( - f^i_1(x))} \big) =  \sum_{i=1}^m \big( \log{\alpha_i} + \log{(C - \alpha_i)} \big).
$$

Hence, the new *centralizer* problem is: 

$$

\begin{align*}
\text{maximize}_{\alpha} \quad & 
\sum_{i=1}^n \alpha_i 
- \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j
+ t \sum_{i=1}^n \Big( \log \alpha_i + \log (C - \alpha_i) \Big) \\
\text{subject to} \quad & 
\sum_{i=1}^n \alpha_i y_i = 0,
\end{align*}

$$

Clearly, for each value of $t$ we have a different solution for this problem, we call them $$\alpha^*(t)$$. Now, one could choose a very small $t$ and just fit Newton's method for equality constrains. However, this is unlikely to work well, because the Hessian is likely to vary very rapidly when the parameters the inequality constraints are close to 0 (as per the Hessian definition above). In that case, the Lipschitz constant of the Hessian is huge and we may need many iterations (to find a region where the gradient is already tiny) to reach the quadratically convergent phase of Newton's method. Hence, the idea is to decrease $t$ "slowly" and initialize the Newton's method in the previous solution, effectibly having a outer loop decreasing $t$ and an inner loop solving the respective equality constrained problems starting at the solution of the previous iteration.

TODO understand exaclty why this really helps, it is not clear to me why the central path is superior to the random initialization per se.

In any case, it is not completely obvious that this should yield the actual optimal for the original inequality constrained problem. Nevertheless, this is actually the case. In particular, for the general minimization case (assuming $f_0$ convex and differentiable, affine equality constraints ($h_i$) and convex inequality constraints ($f_i$)):

$$
\lim_{t \rightarrow 0} f_0(x^*(t)) = f_0(x^*).
$$

When $t$ is very small then the objective value associated to the optimal parameters of the centralizer problem does indeed converge to the objective value of the optimal parameters in the original problem. To see this note the following parallelism between the stationarity condition of the Lagrangian of the centralizer problem and the stationary condition of the original problem. For the centralizer problem:

$$
\nabla f_0(x^*(t)) + t \nabla \phi (x^*(t)) + A^{\top}\nu^*(t) = 0,
$$

for the original problem:

$$
\nabla f_0(x^*) + \sum_{i=1}^m \lambda^*_i \nabla f_i(x^*) + A^{\top}\nu^* = 0.
$$

Since $$\nabla \phi(x) = -\sum_{i=1}^m \frac{1}{f_i(x)} \, \nabla f_i(x)$$ then we have that we can define

$$
\begin{equation}
\lambda^*(t) = -\frac{t}{f_i(x^*(t))},
\label{eq:subst}
\end{equation}
$$

then the dual becomes (assuming $$x^*(t)$$ is feasible)

$$
f_0(x^*) \geq g(\lambda^*(t), \nu^*(t)) 
= f_0(x^*(t)) + \sum_{i=1}^m \lambda_i^*(t) f_i(x^*(t)) + \nu^{*}(t)^\top (Ax^*(t) - b)
= f_0(x^*(t)) - mt.
$$

Hence, $$x^*(t)$$ is only $$mt$$ suboptimal so indeed the intuition is correct and we will recover the optimal solution. Note that, without strict convexity, $x^*$ might not be unique!

Finally, note that we do require to always work with a feasible $x$. This is again not trivial, in fact, it requires an optimization step in itself (solving $$Ax = b$$ for $$x \leq 0$$ is as hard as an LP!). This boils down to solving:

$$
\begin{aligned}
\text{minimize}_{x, s} \quad & s \\
\text{subject to} \quad & Ax = b, \\
& f_i(x) \le s, \quad i = 1,\dots,n.
\end{aligned}
$$

Which is AGAIN an inequality constrained optimization problem. However, it suffices to solve this once with a "big" $$t$$.

### The algorithm pieces for the general case
We derive now the Newton's equations for the general case.

#### Phase I (finding a feasible initial point)
Construct the Lagrangian, for a big $$t$$:

$$
L(x, s, \nu) = s - t\sum_{i = 1}^m \log(s - f_i(x)) + \nu^{\top}(Ax -b).
$$

Gradients and Hessians:

$$
\begin{align*}
\nabla_x L(x, s, \nu) &= \sum_{i=1}^m \frac{t}{s - f_i(x)} \nabla f_i(x) + A^\top \nu, \\
\nabla_s L(x, s, \nu) &= 1 - \sum_{i=1}^m \frac{t}{s - f_i(x)}, \\
\nabla_\nu L(x, s, \nu) &= Ax - b,
\end{align*}
$$

$$
\begin{align*}
\nabla_x^2 L(x, s, \nu) &= \sum_{i=1}^m \left( \frac{t}{s - f_i(x)}\nabla^2 f_i(x) + \frac{t}{(s - f_i(x))^2} \nabla f_i(x) \nabla f_i(x)^\top \right), \\
\nabla_s^2 L(x, s, \nu) &= \sum_{i=1}^m \frac{t}{(s - f_i(x))^2}, \\
\nabla_{xs}^2 L(x, s, \nu) &= - \sum_{i=1}^m \frac{t}{(s - f_i(x))^2} 
\nabla f_i(x),\\
\nabla_{\nu x}^2 L(x, s, \nu) &= A, \\
\nabla_{\nu s}^2 L(x, s, \nu) &= 0, \\
\nabla_{\nu \nu}^2 L(x, s, \nu) &= 0.
\end{align*}
$$

Now, putting all together:

$$
\begin{bmatrix}
\nabla_x^2 L & \nabla_{xs}^2 L & A^\top \\
(\nabla_{xs}^2 L)^\top & \nabla_s^2 L & 0 \\
A & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta \nu
\end{bmatrix}
= -
\begin{bmatrix}
\nabla_x L \\
\nabla_s L \\
\nabla_\nu L
\end{bmatrix}
$$

#### Centering step

$$
L(x, \nu) = f_0(x) - t\sum_{i = 1}^m \log(-f_i(x)) + \nu^{\top}(Ax -b).
$$

$$
\begin{bmatrix}
\nabla^2 f_0(x) + t \nabla^2 \phi(x) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta \nu
\end{bmatrix}
= -
\begin{bmatrix}
\nabla f_0(x) + t \nabla \phi(x) + A^\top \nu \\
Ax - b
\end{bmatrix}
$$

### Log-Barrier Method SVM algorithm

Now we've got everything we need. Bear in mind that we are optimizing the dual of the SVM, so we have a maximization problem! Furthermore, note that there are only 2 inequality constraints and 1 equality constraint acting in $$\alpha$$ as a vector.

First, let's get the derivatives and hessians for the Phase I and central path problems.

The Lagrangian for Phase I is:

$$
L(\alpha, s, \nu) = s - t \sum_{i=1}^n \left( \log(s + \alpha_i) + \log(s + C - \alpha_i) \right) + \nu (\mathbf{y}^\top \alpha).
$$

$$
\begin{align*}
\nabla_\alpha L &= -t \left( \frac{1}{s + \alpha} - \frac{1}{s + C - \alpha} \right) + \nu \mathbf{y} \quad \text{Using elementwise division, } \nu \in \mathbb{R} \text{ (scalar)},\\
\nabla_s L &= 1 - t \sum_{i=1}^n \left( \frac{1}{s + \alpha_i} + \frac{1}{s + C - \alpha_i} \right), \\
\nabla_\nu L &= \mathbf{y}^\top \alpha.
\end{align*}
$$

Let $$d_1 = \frac{1}{(s+\alpha)^2}$$ and $$d_2 = \frac{1}{(C + s -\alpha)^2}$$ elementwise!

$$
\begin{align*}
\nabla_{\alpha \alpha}^2 L &= t \, \text{diag}(d_1 + d_2), \\
\nabla_{ss}^2 L &= t \sum_{i=1}^n (d_1 + d_2), \\
\nabla_{\alpha s}^2 L &= t (d_1 - d_2).
\end{align*}
$$

Finally:

$$
\begin{bmatrix}
t \, \text{diag}(d_1 + d_2) & t(d_1 - d_2) & \mathbf{y} \\
t(d_1 - d_2)^\top & t \sum (d_1 + d_2) & 0 \\
\mathbf{y}^\top & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \alpha \\
\Delta s \\
\Delta \nu
\end{bmatrix}
= -
\begin{bmatrix}
\nabla_\alpha L \\
\nabla_s L \\
\mathbf{y}^\top \alpha
\end{bmatrix}.
$$

The Lagrangian for the Central Path problems is:

$$L(\alpha, \nu) = \mathbf{1}^\top \alpha - \frac{1}{2} \alpha^\top Q \alpha + t \sum_{i=1}^n \Big( \log \alpha_i + \log (C - \alpha_i) \Big) + \nu (\mathbf{y}^\top \alpha),$$

where $$Q_{ij} = y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$. Note the change of sign, the penalty goes negative because we are maximizing.

Again, gradients, Hessians and Newton's matrix (abusing a bit the elementwise operations...):

$$
\begin{align*}
\nabla_{\alpha} L &= \mathbf{1} - Q\alpha + t \left( \frac{1}{\alpha} - \frac{1}{C - \alpha} \right) + \nu \mathbf{y}, \\
\nabla_\nu L &=  \mathbf{y}^\top \alpha,
\end{align*}
$$

Let $$d_1 = \frac{1}{\alpha^2}$$ and $$d_2 = \frac{1}{(C -\alpha)^2}$$ elementwise!

$$
\begin{align*}
\nabla_{\alpha \alpha}^2 L &= -Q - t  \, \text{diag}(d_1 + d_2,) \\
\nabla_{\nu \alpha}^2 L &= \mathbf{y}^\top ,\\
\nabla_{\nu \nu}^2 L &= 0. \\
\end{align*}
$$


And finally

$$
\begin{bmatrix}
-Q - t \, \text{diag}(d_1 + d_2) & \mathbf{y} \\
\mathbf{y}^\top & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \alpha \\
\Delta \nu
\end{bmatrix}
= -
\begin{bmatrix}
\nabla_\alpha L \\
\nabla_\nu L
\end{bmatrix}
$$

Now all the math is ready, the algorithm then is:
<div style="text-align: center;">
<img src="/assets/images/SVMs/LogBarrierSVM.png" alt="LogBarrier image Error" width="600">
</div>

<div style="text-align: center;">
<img src="/assets/images/SVMs/LB.gif" alt="LogBarrier image Error" width="600">
</div>

In the above demo we used RBF kernel for coolness purposes, but that has actually nothing to do with the optimization problem since it is literally just a different $$Q$$.

## Primal-Dual algorithm
The Log-Barrier method can be interpreted as solving a modified KKT system in each inner loop. A "continuous deformation" that indeed converges to the final canonical KKT system. In particular, again for the estandard minimization problem:

$$
\begin{aligned}
Ax &= b, \\
f_i(x) &\le 0, \quad i = 1,\dots,m, \\
\lambda &\ge 0, \\[6pt]

\nabla f_0(x)
+ \sum_{i=1}^m \lambda_i \nabla f_i(x)
+ A^{T}\nu
&= 0, \\[6pt]

-\lambda_i f_i(x) &= t,
\quad i = 1,\dots,m.
\end{aligned}
$$

The only difference is in the complementary slackness condition. In the Log-Barrier method $$\lambda$$ is eliminated via substitution (equation \ref{eq:subst}). The idea now is to directly focus on solving the modified KKT directly, explicitly optimizing $$\lambda$$. This greatly simplifies things. First, now we are allowed to work with not feasible values for primal and dual variables, hence we skip Phase I altogether (which should make this already trivially twice as fast...). Convergence will ensure feasibility, but it is not a requirement at the start! Second, there is only one loop, $$t$$ is updated at the same time as $$\lambda, x, \nu$$.

Let's check the theory again for the standard case and then we fill it in with the SVM problem specifics. The modified KKT equation to solve is:


$$
\begin{bmatrix}
\nabla^2 f_0(x) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(x) & Df(x)^T & A^T \\
-\mathbf{diag}(\lambda)Df(x) & -\mathbf{diag}(f(x)) & 0 \\
A & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta \lambda \\
\Delta \nu
\end{bmatrix}
= -
\begin{bmatrix}
\nabla f_0(x) + Df(x)^{T}\lambda + A^{T}\nu \\
- \operatorname{diag}(\lambda)\, f(x) - t\mathbf{1} \\
Ax - b
\end{bmatrix}

$$

Define $$\hat{\eta}(x, \lambda) = - \sum_{i=1}^m f_i(x)\lambda_i$$. If 

$$
\bigg\|\begin{bmatrix}
\nabla f_0(x) + Df(x)^{T}\lambda + A^{T}\nu \\
- \operatorname{diag}(\lambda)\, f(x) - t\mathbf{1} \\
Ax - b
\end{bmatrix}\bigg\|_2 = 0,
$$

then $$\hat{\eta}(x, \lambda) = mt$$. If $$\lambda, x, \nu$$ satisfies the modified KKT, then $$mt$$ is the duality gap. Nothe that then $$t = \frac{\hat{\eta}(x, \lambda)}{m}$$. And here we have the idea for the algorithm, we are going to decrease the duality gap by decreasing $$t$$ (as before, using $$\mu \in (0, 1)$$) and at the same time forcing feasibility by the minimization of the residual. Now, turning back to SVMs.


Residuals:

$$
\begin{align*}
r_{\text{dual}} &= \mathbf{1} - Q \alpha - \lambda^{\top} 
\begin{bmatrix}
-\mathbf{I} \\
\mathbf{I}
\end{bmatrix}
+ \nu y \\

r_{\text{cent}} &= -\text{diag}(\lambda)\begin{bmatrix}
-\alpha \\
\alpha - C 
\end{bmatrix} -t\mathbf{1} \quad \lambda \text{ is positive and the function values are negative (when feasible)}\\

r_{\text{pri}} &= \mathbf{y}^\top \alpha.
\end{align*}
$$

So the KKT matrix:

$$
\begin{bmatrix}
- Q & \begin{bmatrix}-I \\ I \end{bmatrix}^\top & y \\
-\text{diag}(\lambda) \begin{bmatrix}-I \\ I \end{bmatrix} & \text{diag}\begin{bmatrix}
-\alpha \\
\alpha - C 
\end{bmatrix} & 0 \\
y^\top & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \alpha \\
\Delta \lambda \\
\Delta \nu
\end{bmatrix}
=
-\begin{bmatrix} \mathbf{1} - Q \alpha - \lambda^{\top} 
\begin{bmatrix}
-\mathbf{I} \\
\mathbf{I}
\end{bmatrix}
+ \nu y \\ -\text{diag}(\lambda)\begin{bmatrix}
-\alpha \\
\alpha - C 
\end{bmatrix} -t\mathbf{1}\\ \mathbf{y}^\top \alpha \end{bmatrix}.
$$

Note that, while in the Log-Barrier method we effectively inserted the inequality constraint in the Lagrangian and the continued (sort of) like in equaility constrained optimization, here we are NOT working in finding stationary points of any Lagrangian! Here everything follows from the residual vector.

The line search must ensure that the $$\lambda$$ stays positive and the constraint functions negative. It also ensures that the norm of the residuals decreases sufficiently. From [1] §11.7:
> One iteration of the primal-dual interior-point algorithm is the same as one step of the infeasible Newton method, applied to solving $$\, r_t(x, \lambda, \nu) = 0$$, but modified to ensure $$\, \lambda > 0$$ and $$\, f(x) < 0$$.


Finally, the algorithm:

<div style="text-align: center;">
<img src="/assets/images/SVMs/Primal_dual_svm.png" alt="LogBarrier image Error" width="600">
</div>
<div style="text-align: center;">
<img src="/assets/images/SVMs/PD.gif" alt="LogBarrier image Error" width="600">
</div>

Coding wise, this felt much cleaner than the log-Barrier method...

## ADMM

TODO same, explain and eventually propose an algorithm for SVM. The idea here is to actually leverage some GPU acceleration, since for equality constrained optimization this would
definitaly be possible, like first order methods for exaple... But one need to understand properly what ADMM is first

## Sequential minimal optimization 

TODO same, explain and eventually propose an algorithm for SVM

# SVM in MNIST: Performance comparison!

TODO just put together everything in the code and show the time comparison of the performance in several distinct datasets. Also compare with the off-the-shelve implementations.




---


## References

These notes are a personal synthesis and extensions based on lectures from **Mastermath (NL) Continuous Optimization Course 2025**  along with standard references (below). Any errors are my own.


1. **Boyd, S., & Vandenberghe, L.**  
   *[Convex Optimization](https://stanford.edu/~boyd/cvxbook/)*

2. **Vapnik, V. N.**  
   *[The Nature of Statistical Learning Theory](https://statisticalsupportandresearch.wordpress.com/wp-content/uploads/2017/05/vladimir-vapnik-the-nature-of-statistical-learning-springer-2010.pdf)*


