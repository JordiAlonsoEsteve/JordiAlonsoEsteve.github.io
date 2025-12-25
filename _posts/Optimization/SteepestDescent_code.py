import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from matplotlib.animation import FuncAnimation, PillowWriter

def convex_first_cond():
    # Function and derivative
    def f(y):
        return 0.1 * y**2

    def tangent(y, x):
        return 0.1 * x**2 + 0.2 * x * (y - x)

    # Domain
    y = np.linspace(-6, 6, 400)
    x_vals = np.linspace(-5, 5, 60)

    # Output directory
    frames_dir = "assets/images/SteepestDescent/gif_foc"
    os.makedirs(frames_dir, exist_ok=True)

    frames = []

    for i, x in enumerate(x_vals):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot function
        ax.plot(y, f(y), label=r"$f(y)=0.1y^2$", linewidth=2)

        # Plot tangent
        ax.plot(y, tangent(y, x),
                linestyle="--",
                label=rf"$T_x(y)$ at $x={x:.2f}$")

        # Tangency point
        ax.scatter([x], [f(x)], zorder=3)

        ax.set_xlim(-6, 6)
        ax.set_ylim(-1, 4)
        ax.set_title("First-Order Convexity Condition")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True)

        filename = f"{frames_dir}/frame_{i:03d}.png"
        plt.savefig(filename, dpi=120)
        plt.close()

        frames.append(imageio.imread(filename))

    # Save GIF
    imageio.mimsave("assets/images/SteepestDescent/gif_foc/first_order_convexity.gif",
                    frames,
                    duration=0.08,
                    loop=0)

    print("GIF saved as first_order_convexity.gif")

def gradient_descent():
   # Function value vs iteration for high-dimensional quadratic with exact line search
    np.random.seed(0)


    # Parameters
    n = 100
    max_iter = 50
    x0 = np.random.randn(n)


    # Generate A matrices with specified condition numbers
    def generate_A(n, cond):
        # Eigenvalues from 1 to cond
        vals = np.linspace(1, cond, n)
        Q, _ = np.linalg.qr(np.random.randn(n,n)) # random orthogonal
        return Q @ np.diag(vals) @ Q.T


    A1 = generate_A(n, 2)
    A2 = generate_A(n, 10)
    A3 = generate_A(n, 100)
    A4 = generate_A(n, 500)


    # Gradient descent with exact line search
    def gradient_descent_quadratic(A, x0, max_iter):
        x = x0.copy()
        history = [x.copy()]
        fvals = [0.5*x.T@A@x]
        for _ in range(max_iter):
            grad = A @ x
            alpha = (grad @ grad) / (grad @ A @ grad)
            x = x - alpha * grad
            history.append(x.copy())
            fvals.append(0.5*x.T@A@x)
        return np.array(history), fvals


    hist1, fvals1 = gradient_descent_quadratic(A1, x0, max_iter)
    hist2, fvals2 = gradient_descent_quadratic(A2, x0, max_iter)
    hist2, fvals3 = gradient_descent_quadratic(A3, x0, max_iter)
    hist2, fvals4 = gradient_descent_quadratic(A4, x0, max_iter)


    # Plot function values
    plt.figure(figsize=(10,5))
    plt.plot(fvals1, 'r-', label='cond=2')
    plt.plot(fvals2, 'b-', label='cond=10')
    plt.plot(fvals3, 'g-', label='cond=100')
    plt.plot(fvals4, 'y-', label='cond=500')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent with Exact Line Search on 100D Quadratic')
    plt.yscale('log') # log scale to see differences
    plt.legend()
    plt.grid(True)


    # Add LaTeX function text
    plt.text(2, 10e-16, r'$f(x) =\frac{1}{2} x^T A x$', fontsize=14)


    plt.tight_layout()
    plt.savefig('assets/images/SteepestDescent/gif_gd/my_gd.png')
    plt.close()

def gd_simpler_function():


    # Function definitions
    def f1(x):
        return 0.5*(x[0]**2 + 15*x[1]**2)

    def f2(x):
        return 0.5*(x[0]**2 + 3*x[1]**2)

    # Gradients
    def grad_f1(x):
        return np.array([x[0], 15*x[1]])

    def grad_f2(x):
        return np.array([x[0], 3*x[1]])

    # Exact line search for quadratic
    def exact_line_search_quadratic(x, grad, A):
        return (grad @ grad) / (grad @ A @ grad)

    # Gradient descent function
    def gradient_descent_quadratic_custom(grad_func, A, x0, max_iter=20):
        x = x0.copy()
        history = [x.copy()]
        fvals = [0.5*x.T@A@x]
        for _ in range(max_iter):
            grad = grad_func(x)
            alpha = exact_line_search_quadratic(x, grad, A)
            x = x - alpha * grad
            history.append(x.copy())
            fvals.append(0.5*x.T@A@x)
        return np.array(history), fvals

    # Define A matrices
    A1 = np.array([[1,0],[0,10]])
    A2 = np.array([[1,0],[0,2]])

    # Initial point
    x0 = np.array([10.0, 1.0])

    # Run gradient descent
    hist1, fvals1 = gradient_descent_quadratic_custom(grad_f1, A1, x0, max_iter=30)
    hist2, fvals2 = gradient_descent_quadratic_custom(grad_f2, A2, x0, max_iter=30)

    # Setup grid for contour plots
    xlist = np.linspace(-10, 10, 200)
    ylist = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(xlist, ylist)
    Z1 = 0.5*(X**2 + 15*Y**2)
    Z2 = 0.5*(X**2 + 3*Y**2)

    # Plot setup
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    
    levels = 40
    cs1 = axes[0,0].contourf(X, Y, Z1, levels=levels, cmap='jet')
    cs2 = axes[0,1].contourf(X, Y, Z2, levels=levels, cmap='jet')
    fig.colorbar(cs1, ax=axes[0,0])
    fig.colorbar(cs2, ax=axes[0,1])
    axes[0,0].set_xlim([-10,10]); axes[0,0].set_ylim([-4,4]); axes[0,0].set_title(r'$f(x,y)=\frac{1}{2}(x^2 + 15y^2)$')
    axes[0,1].set_xlim([-10,10]); axes[0,1].set_ylim([-4,4]); axes[0,1].set_title(r'$f(x,y)=\frac{1}{2}(x^2 + 3y^2)$')

    line1, = axes[0,0].plot([], [], 'r-o', linewidth=2, markersize=5)
    line2, = axes[0,1].plot([], [], 'r-o', linewidth=2, markersize=5)
    line_f1, = axes[1,0].plot([], [], 'r-o')
    line_f2, = axes[1,1].plot([], [], 'r-o')

    axes[1,0].set_xlim(0, len(fvals1)); axes[1,0].set_ylim(0, max(fvals1)*1.1); axes[1,0].set_xlabel('Iteration'); axes[1,0].set_ylabel('f(x)'); axes[1,0].set_title('Function value vs iteration'); axes[1,0].grid(True)
    axes[1,1].set_xlim(0, len(fvals2)); axes[1,1].set_ylim(0, max(fvals2)*1.1); axes[1,1].set_xlabel('Iteration'); axes[1,1].set_ylabel('f(x)'); axes[1,1].set_title('Function value vs iteration'); axes[1,1].grid(True)
    fig.suptitle('Gradient Descent Paths with Exact Line Search', fontsize=16, y=0.95)
    # Animation function
    max_frames = len(hist1)
    def update(frame):
        line1.set_data(hist1[:frame+1,0], hist1[:frame+1,1])
        line2.set_data(hist2[:frame+1,0], hist2[:frame+1,1])
        line_f1.set_data(range(frame+1), fvals1[:frame+1])
        line_f2.set_data(range(frame+1), fvals2[:frame+1])
        return line1, line2, line_f1, line_f2

    ani = FuncAnimation(fig, update, frames=max_frames, interval=170, blit=True)  # 30 frames * 170ms ~ 5s

    # Save GIF
    ani.save('assets/images/SteepestDescent/gif_gd/gd_contour_2x2.gif', writer=PillowWriter(fps=6))
    plt.close()

def coordinate_transform():
    def f(x, M, b):
        return x.T@ M @ x + b @ x
    
    def grad_f(x, M, b): # This is normalized euclidean norm steepest descent direction
        g = 2 * M @ x + b
        return g / np.linalg.norm(g)  # unit vector
    
    def Hessian_f(x, M, b):
        return M
    
    def x_bar(x):
        return A @ x
    
    def f_bar(x, M, b):
        return f(np.linalg.inv(A) @ x, M, b)
    
    def grad_f_bar(x, M, b):
        return np.linalg.inv(A).T @ grad_f(x, M, b)  # This is normalized euclidean norm for quadratic norm with P = A
    
    def Hessian_f_bar(x, M, b):
        return np.linalg.inv(A).T @ Hessian_f(x, M, b) @ np.linalg.inv(A)
    
    M = np.array([[10, 2],
                 [10, 40]])

    # Example usage
    theta = np.pi / 4  # 45 degrees
    A = np.array([[1, 0],[0, 1.5]])
    b = np.array([32, -25])
    
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    # --- Evaluate f on grid ---
    Z = np.zeros_like(X)
    Z_bar = np.zeros_like(X)
    Xb = np.zeros_like(X)  # transformed coordinates
    Yb = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = f(vec, M, b)

            xb_vec = x_bar(vec)        # transformed point
            Z_bar[i,j] = f_bar(xb_vec, M, b)  # same as f(vec)
            Xb[i,j], Yb[i,j] = xb_vec  # store transformed coordinates

    # --- Choose a point ---
    point = np.array([1.0, 0.5])          # in x-space
    point_bar = x_bar(point)              # corresponding point in bar-x

    # --- Plot heatmaps / contours ---
    fig, axes = plt.subplots(1, 2, figsize=(13,8))

    # Determine limits based on original and transformed grids
    x_min = X.min()
    x_max = X.max()
    y_min = Y.min()
    y_max = Y.max()

     # --- Gradient vectors (unit) ---
    grad_pt = -grad_f(point, M, b)
    grad_pt_bar = -grad_f_bar(point_bar, M, b)
    # --- Original f(x) ---
    ax = axes[0]
    cont1 = ax.contour(X, Y, Z, levels=25, cmap="jet")
    ax.plot(point[0], point[1], 'ko', markersize=6)
    ax.quiver(point[0], point[1], grad_pt[0], grad_pt[1], color='r', scale=5, label='GD normalized SD direction')
    ax.set_title(f"Original function $f(x) = x^T M x + bx $ \n $\kappa = {np.linalg.cond(Hessian_f(x, M, b)):.2f}$", fontsize=14)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.legend()

    # --- Transformed f_bar(x_bar) ---
    ax = axes[1]
    cont2 = ax.contour(Xb, Yb, Z_bar, levels=25, cmap="jet")
    ax.plot(point_bar[0], point_bar[1], 'ko', markersize=6)
    ax.quiver(point_bar[0], point_bar[1], grad_pt_bar[0], grad_pt_bar[1], color='r', scale=5)
    ax.set_title(f"Transformed coordinates \n $\kappa = {np.linalg.cond(Hessian_f_bar(x, M, b)):.2f}$", fontsize=14)
    ax.set_xlabel("$\overline{x}_1$")
    ax.set_ylabel("$\overline{x}_2$")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.legend()

    plt.tight_layout()
    plt.savefig("assets/images/SteepestDescent/Coordinate_transform.png")


#gd_simpler_function()
coordinate_transform()