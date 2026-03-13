import numpy as np
import matplotlib.pyplot as plt
import time 

# -------------------------
# Problem setup
# -------------------------

n = 1000     # dimension of x
m = 100   # number of constraints

# Positive definite quadratic objective
Q = np.diag(np.linspace(1.0, 3.0, n))
q = np.random.normal(size=(n))

def f(x):
    return 0.5 * x @ Q @ x + q @ x

def grad_f(x):
    return Q @ x + q

def hess_f(x):
    return Q

# Linear constraints Ax = b
A = np.random.randn(m, n)
assert np.linalg.matrix_rank(A) == m

# Projection matrix onto ker(A)
P = np.eye(n) - A.T @ np.linalg.inv(A @ A.T) @ A

# Initial feasible point
b = np.random.normal(size=m, scale=1)
x0 = np.linalg.pinv(A) @ b

assert np.allclose(A @ x0, b)  # fix assertion

# -------------------------
# Algorithms
# -------------------------

def newton_kkt(x0, max_iter=50, tol=1e-3):
    """Newton's method for quadratic objective with linear constraints"""
    x = x0.copy()
    lam = np.zeros(A.shape[0])
    xs = []

    start = time.time()
    # Precompute KKT matrix since Q and A are constant
    KKT = np.block([
        [Q, A.T],
        [A, np.zeros((A.shape[0], A.shape[0]))]
    ])
    
    for iter in range(max_iter):
        # Residual
        r1 = grad_f(x) + A.T @ lam
        r2 = A @ x - b
        res = np.concatenate([r1, r2])
        
        # Solve KKT * delta = residual
        delta = np.linalg.solve(KKT, res)
        dx = delta[:n]
        dlam = delta[n:]
        
        x -= dx
        lam -= dlam
        
        xs.append(x.copy())
        if np.linalg.norm(dx) < tol and np.linalg.norm(dlam) < tol:
            break
    
    end = time.time()
    print(f"Newton KKT method took {end - start:.4f} seconds, final ||dx||={np.linalg.norm(dx):.2e}")
    return np.array(xs)

def gd_norm_gradient(x0, lam0, lr=0.001, steps=15000):
    """Gradient descent on 1/2 ||F||^2"""
    x = x0.copy()
    lam = lam0.copy()
    xs = []

    start = time.time()  # start timer
    for _ in range(steps):
        grad_x = hess_f(x) @ (grad_f(x) + A.T @ lam) + A.T @ (A @ x - b)
        grad_lam = A @ grad_f(x) + A @ A.T @ lam

        x -= lr * grad_x
        lam -= lr * grad_lam
        if np.linalg.norm(grad_x) +  np.linalg.norm(grad_lam) < 1e-3:
            break
        xs.append(x.copy())
    end = time.time()  # end timer
    print(f"GD on ||F||² took {end - start:.4f} seconds, {np.linalg.norm(np.linalg.norm(grad_x) +  np.linalg.norm(grad_lam))}")
    return np.array(xs)


def projected_gd(x0, lr=0.001, steps=15000):
    """Projected Gradient Descent"""
    x = x0.copy()
    xs = []

    start = time.time()
    for _ in range(steps):
        g = grad_f(x)
        x -= lr * (P @ g)
        xs.append(x.copy())
        if np.linalg.norm((P @ g)) < 1e-3:
            break
    end = time.time()
    print(f"Projected GD took {end - start:.4f} seconds, {np.linalg.norm((P @ g))}")
    return np.array(xs)

# -------------------------
# Run experiments
# -------------------------

lam0 = np.zeros(m)
# -------------------------
# Run experiments and record time
# -------------------------
start = time.time()
xs_kkt = gd_norm_gradient(x0, lam0)
time_kkt = time.time() - start

start = time.time()
xs_pgd = projected_gd(x0)
time_pgd = time.time() - start

f_kkt = [f(x) for x in xs_kkt]
f_pgd = [f(x) for x in xs_pgd]

viol_kkt = [np.linalg.norm(A @ x - b) for x in xs_kkt]
viol_pgd = [np.linalg.norm(A @ x - b) for x in xs_pgd]

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(f_kkt, label="GD on ||F||²", linewidth=2)
plt.plot(f_pgd, label="Projected GD", linewidth=2)
plt.title("Objective value", fontsize=12)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("f(x)", fontsize=10)
plt.legend(fontsize=10)

# Constraint violation
plt.subplot(1, 2, 2)
plt.plot(viol_kkt, label="GD on ||F||²", linewidth=2)
plt.plot(viol_pgd, label="Projected GD", linewidth=2)
plt.title("Constraint violation ||Ax - b||", fontsize=12)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("||Ax - b||_2", fontsize=10)
plt.legend(fontsize=10)

# Overall title with runtime
plt.suptitle(f"Runtime: GD on ||F||² = {time_kkt:.2f}s, PGD = {time_pgd:.2f}s", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

# Save figure in high resolution (300 dpi) and vector format
plt.savefig("assets/images/SteepestDescent/constrained_optimization_comparison_gd.png", dpi=300, bbox_inches='tight')
# Or for PNG: plt.savefig("constrained_optimization_comparison.png", dpi=300, bbox_inches='tight')

xs_newton = newton_kkt(x0)
f_newton = [f(x) for x in xs_newton]
viol_newton = [np.linalg.norm(A @ x - b) for x in xs_newton]

# -------------------------
# Plot Newton separately
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(f_newton, label="Newton KKT", linewidth=2, color='green')
plt.title("Objective value (Newton)", fontsize=12)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("f(x)", fontsize=10)
plt.legend(fontsize=10)

plt.subplot(1,2,2)
plt.plot(viol_newton, label="Newton KKT", linewidth=2, color='green')
plt.title("Constraint violation ||Ax - b|| (Newton)", fontsize=12)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("||Ax - b||", fontsize=10)
plt.legend(fontsize=10)

plt.suptitle(f"Newton KKT method runtime", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("assets/images/SteepestDescent/constrained_optimization_newton.png", dpi=300, bbox_inches='tight')
plt.show()
