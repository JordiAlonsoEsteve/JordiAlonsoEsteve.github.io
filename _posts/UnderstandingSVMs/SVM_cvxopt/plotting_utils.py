
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def create_svm_gif_linear(x, y, alphas_list, value_history, C, filename="svm_evolution.gif"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pre-calculate axis limits
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        alpha = alphas_list[frame]
        
        # --- Plot 1: Decision Boundary ---
        # Calculate w and b for current alpha
        w = np.sum((alpha * y)[:, np.newaxis] * x, axis=0)
        # Using the average of SVs for b or your nu if you saved it
        # Here we approximate b for visualization purposes
        sv_idx = np.where((alpha > 1e-4) & (alpha < C - 1e-4))[0]
        if len(sv_idx) > 0:
            b = np.mean(y[sv_idx] - np.dot(x[sv_idx], w))
        else:
            b = 0

        # Plot Data
        ax1.scatter(x[y==1, 0], x[y==1, 1], c='blue', edgecolors='k', label='+1')
        ax1.scatter(x[y==-1, 0], x[y==-1, 1], c='red', edgecolors='k', label='-1')
        
        # Decision Surface
        Z = (w[0] * xx + w[1] * yy + b)
        ax1.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
                    linestyles=['--', '-', '--'], alpha=0.8)
        
        ax1.set_title(f"Iteration {frame} (t update)")
        ax1.legend()

        # --- Plot 2: Objective Function ---
        ax2.plot(value_history[:frame+1], color='green', marker='o', markersize=4)
        ax2.set_title("Dual Objective Value (Maximizing)")
        ax2.set_xlabel("Outer Iteration")
        ax2.set_ylabel("f(alpha)")
        
    ani = FuncAnimation(fig, update, frames=len(alphas_list), repeat=True)
    
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close()
    print(f"GIF saved as {filename}")

def create_svm_gif_kernelized(X, y, alphas_list, value_history, C, kernel, filename, gamma=1.0,):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Define grid for evaluation
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Pre-compute Grid Kernel for speed
    K_grid = kernel(X, grid = grid)
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        alpha = alphas_list[frame]
        y_alpha = alpha * y
        
        # Calculate b (Bias) using Support Vectors
        sv_mask = (alpha > 1e-4) & (alpha < C - 1e-4)
        if np.any(sv_mask):
            K_sv = kernel(X)
            b = np.mean(y - np.dot(K_sv, y_alpha))
        else:
            b = 0

        # Decision Function Values
        Z = (K_grid @ y_alpha + b).reshape(xx.shape)

        # 1. Background Shading (Heatmap of confidence)
        # Use a diverging colormap like 'RdBu' (Red = Negative, Blue = Positive)
        im = ax1.pcolormesh(xx, yy, Z, cmap='RdBu', shading='auto', alpha=0.3, vmin=-2, vmax=2)
        
        # 2. Add Contour Lines with Labels
        # We explicitly label -1, 0, and 1
        contours = ax1.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
                            linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
        ax1.clabel(contours, inline=True, fontsize=10, fmt={-1: '-1', 0: '0', 1: '+1'})
        
        # 3. Scatter Data and Support Vectors
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50, zorder=5, vmin=-1.5, vmax=1.5)
        
        # Highlight SVs (where alpha > 0)
        sv_idx = np.where(alpha > 1e-3)[0]
        ax1.scatter(X[sv_idx, 0], X[sv_idx, 1], s=130, facecolors='none', edgecolors='lime', 
                    linewidths=2, label=f'SVs (Count: {len(sv_idx)})', zorder=4)
        
        ax1.set_title(f"RBF SVM: Iteration {frame}")
        ax1.legend(loc='upper right')

        # 4. Objective Plot
        ax2.plot(value_history[:frame+1], color='indigo', marker='o', markersize=3)
        ax2.set_title("Objective function")
        ax2.grid(True, linestyle='--', alpha=0.6)

    # Note: Adding a colorbar once outside update is tricky with FuncAnimation
    # Usually we add it to the figure once
    # cbar = fig.colorbar(im, ax=ax1)
    # cbar.set_label('Confidence Score f(x)')

    ani = FuncAnimation(fig, update, frames=len(alphas_list), interval=100)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.show()