import numpy as np

### Phase I machinery

def PI_Lagrangian(alpha, s, nu, t, y, C) -> float:
    # Barrier
    phi_alpha = t*np.sum(np.log(s + alpha) + np.log(s + C - alpha))
    # equality constraint "penalty"
    nu_penalty = nu * y @ alpha

    return s - phi_alpha + nu_penalty

def PI_L_grad_alpha(alpha, s, nu, t, y, C) -> np.array:
    return -t *( (1/(s + alpha)) - (1/(s+C-alpha)) ) + nu * y

def PI_L_grad_s(alpha, s, nu, t, y, C)->  float: # s is an scalar
    return 1 - t * np.sum( (1/(s + alpha)) + (1/(s+C-alpha)) )

def PI_L_grad_nu(alpha, s, nu, t, y, C):
    return y @ alpha

def PI_L_hessian_alpha2(alpha, s, nu, t, y, C):
    return t * np.diag(1/(s + alpha)**2 + 1/(C + s - alpha)**2)

def PI_L_hessian_s2(alpha, s, nu, t, y, C):
    return t * np.sum(1/(s + alpha)**2 + 1/(C + s - alpha)**2)

def PI_L_hessian_alpha_s(alpha, s, nu, t, y, C):
    return t * (1/(s + alpha)**2 - 1/(C + s - alpha)**2)

### Central path machinery

def CP_Lagrangian(alpha, Q, nu, t, y, C):
    qt = 0.5 *alpha.T @ Q @ alpha
    barrier = t * np.sum(np.log(alpha) + np.log(C - alpha))
    eq_c = nu * y @ alpha

    return np.sum(alpha) - qt + barrier + eq_c

def CP_L_grad_alpha(alpha, Q, nu, t, y, C):
    return 1 - Q @ alpha + t* (1/ alpha - 1/(C - alpha)) + nu * y

def CP_L_grad_nu(alpha, Q, nu, t, y, C):
    return y @ alpha

def CP_L_hessian_alpha2(alpha, Q, nu, t, y, C):
    return -Q - t*np.diag(1/(alpha)**2 + 1/(C- alpha)**2)

def CP_L_hessian_nu_alpha(alpha, Q, nu, t, y, C):
    return y.T

def f_0(alpha, Q, nu, t, y, C):
    barrier = t * np.sum(np.log(alpha) + np.log(C - alpha))
    return np.sum(alpha) - 0.5 *alpha.T @ Q @ alpha + barrier

def g_0(alpha, Q, nu, t, y, C):
    return 1 - Q @ alpha + t* (1/ alpha - 1/(C - alpha)) 



### loop functionality Phase I
def _solve_NS_phase_I(alpha, s, nu, t, y, C):
    g_a = PI_L_grad_alpha(alpha, s, nu, t, y, C)
    g_s = PI_L_grad_s(alpha, s, nu, t, y, C)
    eq_resid = PI_L_grad_nu(alpha, s, nu, t, y, C) # This is effectively grad_nu
    # Important, note that this is not necessarily 0 if the staring point is not feasible, which is the
    # default!
    
    # Hessians (Matrix Components)
    # Note: alpha2 returns the diagonal matrix, but we only need the diagonal vector
    H_aa_diag = np.diag(PI_L_hessian_alpha2(alpha, s, nu, t, y, C))
    H_ss = PI_L_hessian_s2(alpha, s, nu, t, y, C)
    g_as = PI_L_hessian_alpha_s(alpha, s, nu, t, y, C)

    n = len(alpha) # number of observations!

    KKT = np.zeros((n + 2, n + 2)) # s and nu are escalars

    np.fill_diagonal(KKT[:n, :n], H_aa_diag + 1e-12)
    
    # Fill Block Row 1 (Interactions with alpha)
    KKT[:n, n] = g_as         # Connection to s
    KKT[:n, n+1] = y          # Connection to nu
    
    # Fill Block Row 2 (Interactions with s)
    KKT[n, :n] = g_as         # Symmetry
    KKT[n, n] = H_ss          # Second derivative wrt s
    
    # Fill Block Row 3 (Equality constraint)
    KKT[n+1, :n] = y          # y^T * delta_alpha
    
    rhs = -np.concatenate([g_a, [g_s], [eq_resid]])
    
    try:
        # solve() is generally more stable than inv()
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        # Fallback for ill-conditioned matrices near the boundary
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        
    delta_alpha = sol[:n]
    delta_s = sol[n]
    delta_nu = sol[n+1]
    
    return delta_alpha, delta_s, delta_nu

def PI_residual(alpha, s, nu, t, y, C):
    g_a = PI_L_grad_alpha(alpha, s, nu, t, y, C)
    g_s = PI_L_grad_s(alpha, s, nu, t, y, C)
    eq  = PI_L_grad_nu(alpha, s, nu, t, y, C)
    return np.concatenate([g_a, [g_s], [eq]])

def PI_merit(alpha, s, nu, t, y, C):
    r = PI_residual(alpha, s, nu, t, y, C)
    return 0.5 * np.dot(r, r)

def backtracking_line_search_PI_residual(
    alpha, s, nu,
    delta_alpha, delta_s, delta_nu,
    t, y, C, beta=0.5
):
    lr = 1.0
    phi0 = PI_merit(alpha, s, nu, t, y, C)

    while True:
        a_new = alpha + lr * delta_alpha
        s_new = s     + lr * delta_s
        nu_new = nu   + lr * delta_nu

        # domain check (barrier feasibility)
        if np.any(a_new + s_new <= 0) or np.any(C - a_new + s_new <= 0):
            lr *= beta
            continue

        phi_new = PI_merit(a_new, s_new, nu_new, t, y, C)

        if phi_new <= phi0:
            return lr

        lr *= beta

def phase_I(alpha, y, C, s, nu, t = 0.1, epsilon = 1e-8):
    if np.abs(y @ alpha) < 1e-10 and (alpha > 0).all() and (alpha < C).all():
        print("Initial values are already feasible!")
        return alpha, s, nu
    while True:
        while True:
            delta_alpha, delta_s, delta_nu =_solve_NS_phase_I(alpha, s, nu, t, y, C)
            lr = backtracking_line_search_PI_residual(
                alpha, s, nu,
                delta_alpha, delta_s, delta_nu,
                t, y, C
            )
            alpha = alpha + lr * delta_alpha
            s =  s + lr * delta_s
            nu = nu + lr * delta_nu

            if PI_merit(alpha, s, nu, t, y, C) < epsilon:
                break
        
        if 2*t < epsilon:
            break
        t = t * 0.01
    
    print(f"Phase I completed, t = {t}")            
    if np.abs(y @ alpha) < 1e-10 and (alpha > 0).all() and (alpha < C).all():
        print("Feasible solution encountered")
    else:
        print("No feasible solution encountered")
    return alpha, s, nu

## Phase II loop functionality

def _solve_NS_phase_II(alpha, Q, nu, t, y, C,):
    g_a = CP_L_grad_alpha(alpha, Q, nu, t, y, C)
    g_nu = CP_L_grad_nu(alpha, Q, nu, t, y, C)
    
    # Hessians (Matrix Components)
    # Note: alpha2 returns the diagonal matrix, but we only need the diagonal vector
    H_aa = CP_L_hessian_alpha2(alpha, Q, nu, t, y, C)
    H_nu_a = CP_L_hessian_nu_alpha(alpha, Q, nu, t, y, C)

    n = len(alpha) # number of observations!

    KKT = np.zeros((n + 1, n + 1)) # s and nu are escalars

    KKT[:n, :n] = H_aa + 1e-12
    
    # Fill Block Row 1 (Interactions with alpha)
    KKT[:n, n] = H_nu_a         # Connection to nu
    
   
    KKT[n, :n] = H_nu_a        # y^T * delta_alpha
    
    rhs = -np.concatenate([g_a, [g_nu]])
    
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        
    delta_alpha = sol[:n]
    delta_nu = sol[n]
    
    return delta_alpha, delta_nu

def backtracking_armijos(alpha, Q, nu, t, y, C, delta_alpha, q = 0.5, b = 0.5):
    lr = 1
    while True:
        if f_0(alpha + lr * delta_alpha, Q,  nu, t, y, C) >= f_0(alpha, Q,  nu, t, y, C) + q * lr * (g_0(alpha, Q,  nu, t, y, C).T @ delta_alpha):
            break
        else:
            lr = lr * b
    return lr

def construct_Q(x, y):
    K = x @ x.T
    return np.outer(y, y) * K


def rbf_kernel_matrix(X, gamma=1.0):
    """Computes the Gaussian RBF Kernel: K(x,y) = exp(-gamma * ||x-y||^2)"""
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

def construct_Q_rbf(X, y, gamma=1.0):
    K = rbf_kernel_matrix(X, gamma)
    Q = np.outer(y, y) * K
    Q += 1e-7 * np.eye(len(y))  # Regularization for PD Hessian
    return Q, K


def phase_II(x, y, C, alpha, nu,  t = 0.1, epsilon = 1e-7):
    alphas_list = []
    value_function = []
    nu_list = []
    assert np.abs(y @ alpha) < 1e-10 and (alpha > 0).all() and (alpha < C).all(), "Values are not feasible!"
    #Q = construct_Q(x, y)
    Q = construct_Q_rbf(x, y)[0]
    while True:
        while True:
            delta_alpha, delta_nu =_solve_NS_phase_II(alpha, Q, nu, t, y, C)
            lr = backtracking_armijos(alpha, Q, nu, t, y, C, delta_alpha)

            nu = nu + lr * delta_nu
            alpha = alpha + lr * delta_alpha
            nu_list.append(nu)
            alphas_list.append(alpha)
            value_function.append(f_0(alpha, Q, nu, 0, y, C))

            g_a = CP_L_grad_alpha(alpha, Q, nu, t, y, C)
            g_nu = CP_L_grad_nu(alpha, Q, nu, t, y, C)
            if  np.linalg.norm(np.concatenate([g_a, [g_nu]]))  < 1e-5: # Just check the gradient is close to 0
                # This is the gradient of the of the Lagrangian!
                break
        
        if 2*t < epsilon:
            break

        t = t * 0.01
    
    print(f"Phase II completed, t = {t}")            
    
    return alpha, alphas_list, value_function, nu_list

if __name__ == "__main__":
    from test_utils import test_derivative
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    def test_phaseI():

        ## TEST PHASE I
        # 1. Setup dimensions and constants
        n = 60
        C = 10.0
        t = 1
        nu = 1.0
        s = 100.0 # In Phase I, s must be large enough so alpha is feasible
        # 2. Generate feasible alpha (0 < alpha < C)
        # np.random.uniform is better here than normal to ensure we stay in (0, C)
        #alpha = np.random.uniform(0, C, size=(n,))\
        alpha = np.zeros(shape=(n, ))
        # 3. Generate labels
        y = np.random.choice([-1, 1], size=(n,))

        # 4. Define the 'other' dictionary 
        # These keys must match the argument names in your PI_Lagrangian function
        input = {
            'alpha': alpha,
            'y': y,
            'C': C,
            't': t,
            'nu': nu,
            's': s
        }
        

        # 5. Run the test
        # We pass the analytical gradient function as the second argument
        error1 = test_derivative(PI_Lagrangian, PI_L_grad_alpha, input, variable_of_interest= 'alpha')
        error2 = test_derivative(PI_Lagrangian, PI_L_grad_s, input, variable_of_interest= 's')
        error3 = test_derivative(PI_Lagrangian, PI_L_grad_nu, input, variable_of_interest= 'nu')
        error4 = test_derivative(PI_L_grad_alpha, PI_L_hessian_alpha2, input, variable_of_interest= 'alpha')
        error5 = test_derivative(PI_L_grad_s, PI_L_hessian_s2, input, variable_of_interest= 's')
        error6 = test_derivative(PI_L_grad_alpha, PI_L_hessian_alpha_s, input, variable_of_interest= 's')
    
        assert (np.array([error1, error2, error3, error4, error5, error6]) < 1e-2).all()

        alpha, s, nu = phase_I(alpha, y, C, s=s, t=t , nu = nu)
        assert np.abs(y@alpha) <= 1e-10
        assert np.sum(alpha < C) == n
        assert np.sum(alpha > 0) == n
        assert s < 0


    
    def generate_test_data(n=10, p=5):
        # 1. Random features (X) and labels (y)
        X = np.random.randn(n, p)
        y = np.random.choice([-1, 1], size=n)
        
        # 2. Linear Kernel: K = X @ X.T
        K = X @ X.T
        
        # 3. Q_ij = y_i * y_j * K_ij
        # This is equivalent to np.outer(y, y) * K
        Q = np.outer(y, y) * K
        
        # Add a tiny bit of jitter to the diagonal for numerical stability (PSD -> PD)
        Q += 1e-8 * np.eye(n)
        
        return Q, y, X
    
    def generate_test_data_2d(n_per_class=10, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 1. Generate two clusters in 2D
        mean_pos = np.array([2, 2])
        mean_neg = np.array([-2, -2])
        cov = 3 * np.eye(2)  # moderate spread
        
        X_pos = np.random.multivariate_normal(mean_pos, cov, size=n_per_class)
        X_neg = np.random.multivariate_normal(mean_neg, cov, size=n_per_class)
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*n_per_class + [-1]*n_per_class)
        
        # 2. Linear kernel
        K = X @ X.T
        
        # 3. Q_ij = y_i * y_j * K_ij
        Q = np.outer(y, y) * K
        
        # Add tiny jitter for numerical stability
        Q += 1e-8 * np.eye(2*n_per_class)
        
        return Q, y, X
        
    n = 120
    C = 5.0
    t = 1
    nu = 1.0
    s = 100.0 # In Phase I, s must be large enough so alpha is feasible (essentially larger than C)
    alpha = np.zeros(shape=(n, ))
    #Q, y, x = generate_test_data(n, p = 2)
    Q, y, x = generate_test_data_2d(int(n/2))
    alpha, _, nu = phase_I(alpha, y, C, s=s, t=t , nu = nu)

    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    
    input = {
        'alpha': alpha,
        'y': y,
        'C': C,
        't': t,
        'nu': nu,
        'Q': Q,
    }

    #error = test_derivative(CP_Lagrangian, CP_L_grad_alpha, input, variable_of_interest= 'alpha')
    #print(f"CP grad alpha Relative Error: {error}")

    #error = test_derivative(CP_Lagrangian, CP_L_grad_nu, input, variable_of_interest= 'nu')
    #print(f"CP grad nu Relative Error: {error}")

    #error = test_derivative(CP_L_grad_alpha, CP_L_hessian_alpha2, input, variable_of_interest= 'alpha')
    #print(f"CP hessian alpha**2 Relative Error: {error}")

    #error = test_derivative(CP_L_grad_nu, CP_L_hessian_nu_alpha, input, variable_of_interest= 'alpha')
    #print(f"CP hessian alpha**2 Relative Error: {error}")
    alpha, alphas_list, value_function, nu_list= phase_II(x=x, y=y, C= C, alpha= alpha, nu= nu)


    def create_svm_gif(x, y, alphas_list, value_history, filename="svm_evolution.gif"):
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

    #create_svm_gif(x, y, alphas_list, value_function)

    def plot_svm_evolution(X, y, alphas_list, value_history, gamma=1.0, C=10.0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Define grid for evaluation
        x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
        y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Pre-compute Grid Kernel for speed
        sq_grid = np.sum(grid**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(grid, X.T)
        K_grid = np.exp(-gamma * sq_grid)

        def update(frame):
            ax1.clear()
            ax2.clear()
            
            alpha = alphas_list[frame]
            y_alpha = alpha * y
            
            # Calculate b (Bias) using Support Vectors
            sv_mask = (alpha > 1e-4) & (alpha < C - 1e-4)
            if np.any(sv_mask):
                K_sv = np.exp(-gamma * (np.sum(X[sv_mask]**2, axis=1).reshape(-1, 1) + 
                                    np.sum(X**2, axis=1) - 2 * np.dot(X[sv_mask], X.T)))
                b = np.mean(y[sv_mask] - np.dot(K_sv, y_alpha))
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
            ax1.clabel(contours, inline=True, fontsize=10, fmt={-1: 'Margin (-1)', 0: 'Boundary (0)', 1: 'Margin (+1)'})
            
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
            ax2.set_title("Dual Objective (L_dual)")
            ax2.grid(True, linestyle='--', alpha=0.6)

        # Note: Adding a colorbar once outside update is tricky with FuncAnimation
        # Usually we add it to the figure once
        # cbar = fig.colorbar(im, ax=ax1)
        # cbar.set_label('Confidence Score f(x)')

        ani = FuncAnimation(fig, update, frames=len(alphas_list), interval=150)
        ani.save("svm_rbf_final.gif", writer=PillowWriter(fps=6))
        plt.show()

    plot_svm_evolution(x, y, alphas_list, value_function)
