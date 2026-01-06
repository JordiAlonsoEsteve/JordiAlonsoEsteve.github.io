import numpy as np

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

def construct_Q(x, y):
    K = x @ x.T
    return np.outer(y, y) * K

def rbf_kernel_matrix(X, gamma=1.0, grid = None):
    """Computes the Gaussian RBF Kernel: K(x,y) = exp(-gamma * ||x-y||^2)"""
    if grid is None:
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    else:
        sq_dists= np.sum(grid**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(grid, X.T)
    return np.exp(-gamma * sq_dists)

def construct_Q_rbf(X, y, gamma=1.0):
    K = rbf_kernel_matrix(X, gamma)
    Q = np.outer(y, y) * K
    Q += 1e-7 * np.eye(len(y))  # Regularization for PD Hessian
    return Q