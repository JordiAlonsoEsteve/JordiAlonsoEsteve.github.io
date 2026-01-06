import numpy as np
from data_utils import generate_test_data, generate_test_data_2d, construct_Q, construct_Q_rbf, rbf_kernel_matrix
from plotting_utils import create_svm_gif_linear, create_svm_gif_kernelized
from tqdm import tqdm

def L_not_equal(C, alpha_i, alpha_j):
    return np.max((0, alpha_j - alpha_i))

def H_not_equal(C, alpha_i, alpha_j):
    return np.min((C, C+ alpha_j - alpha_i))

def L_equal(C, alpha_i, alpha_j):
    return np.max((0, alpha_j + alpha_i - C))

def H_equal(C, alpha_i, alpha_j):
    return np.min((C, alpha_j + alpha_i))

def update_w(w_old, delta_i, delta_j, yj, yi, xi, xj):
    return w_old + delta_i*yi*xi + delta_j*yj*xj

def update_b(b_old, Ei, Ej, yi, yj, alpha_i, alpha_j, alpha_i_old, alpha_j_old, x, i, j, C, sq_norm_X, gamma = 1.0):
    # 1. Calculate necessary kernel elements on the fly
    # K(i,i) and K(j,j) are 1.0 for RBF
    kii = 1.0
    kjj = 1.0
    
    dist_ij = sq_norm_X[i] + sq_norm_X[j] - 2 * np.dot(x[i], x[j])
    kij = np.exp(-gamma * dist_ij)
    
    delta_i = alpha_i - alpha_i_old
    delta_j = alpha_j - alpha_j_old

    # 3. Calculate candidate b values
    b1 = b_old - Ei - yi * delta_i * kii - yj * delta_j * kij
    b2 = b_old - Ej - yi * delta_i * kij - yj * delta_j * kjj
    
    # 4. Selection Logic
    if 0 < alpha_i < C: # C is your soft-margin hyperparameter
        return b1
    elif 0 < alpha_j < C:
        return b2
    else:
       # If both are at bounds (0 or C), the new b is the average
        return (b1 + b2) / 2.0

def get_eta(i, j, X, sq_norm_X, gamma = 1.0):
    """
    i, j: indices of the two samples being optimized
    X: full training matrix (60k, D)
    gamma: RBF parameter
    sq_norm_X: pre-calculated squared norms for all X
    """
    # 1. K(i, i) and K(j, j) for RBF kernel are ALWAYS 1.0
    # Because exp(-gamma * ||x - x||^2) = exp(0) = 1
    kii = 1.0
    kjj = 1.0
    
    # 2. Calculate K(i, j)
    dist_ij = sq_norm_X[i] + sq_norm_X[j] - 2 * np.dot(X[i], X[j])
    kij = np.exp(-gamma * dist_ij)
    
    # 3. Final eta
    eta_val = kii + kjj - 2 * kij
    
    # SMO logic: eta should be > 0. If it's not, the points are too similar.
    return eta_val

def update_alpha_j(alpha_j, Error_i, Error_j, yj, eta):
    return alpha_j + ((yj * (Error_i - Error_j)) / eta)

def update_alpha_i(delta_j, alpha_i, yi, yj):
    return alpha_i + yi*yj*(delta_j)

def objective_function(alpha, Q):
    return np.sum(alpha) - 0.5 * alpha.T @ Q @ alpha


def SVM_output_k(k, X, y, alpha, b, sv_idx, sq_norm_X, gamma = 1.0):
    """
    k: The index of the training point we are interested
    sq_norm_X: The pre-calculated array of all squared norms
    
    K(xi​,xj​)=exp(-gamma∥xi​-xj​∥2)

    ∥xi​-xj​∥2=∥xi​∥2+∥xj​∥2-2xiT​xj​
    
    """
    # 1. Get the support vector data
    X_sv = X[sv_idx]
    y_sv = y[sv_idx]
    alpha_sv = alpha[sv_idx]
    
    # 2. Grab the specific pre-calculated norms we need
    sq_norm_sv = sq_norm_X[sv_idx]
    sq_norm_k = sq_norm_X[k]
    
    # 3. Compute kernel values between training point 'k' and all support vectors
    # We slice X[k] to ensure it's the specific vector we need
    # sq_dists shape: (len(sv_idx),)
    sq_dists = sq_norm_sv + sq_norm_k - 2 * np.dot(X_sv, X[k])
    k_vector = np.exp(-gamma * sq_dists)
    
    # 4. Decision value u_k
    u_k = np.dot(k_vector, (alpha_sv * y_sv)) + b
    return u_k

def compute_dual_objective(alpha, y, X, sq_norm_X, gamma = 1.0):
    # To track progress, we calculate: sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K_ij)
    # For debugging/tracking on small sets, we can compute the whole kernel
    # K_ij = exp(-gamma * (||xi||^2 + ||xj||^2 - 2<xi, xj>))
    dist_matrix = sq_norm_X.reshape(-1, 1) + sq_norm_X.reshape(1, -1) - 2 * np.dot(X, X.T)
    K = np.exp(-gamma * dist_matrix)
    
    term1 = np.sum(alpha)
    # Use matrix form: 0.5 * (alpha*y)^T @ K @ (alpha*y)
    ay = alpha * y
    term2 = 0.5 * ay.T @ K @ ay
    return term1 - term2

def compute_all_errors(x, y, alpha, b, sv_idx, sq_norm_X, gamma = 1.0):
    """
    Computes E = u - y for all N points using matrix multiplication.
    This avoids the Python loop and uses BLAS for speed.
    """
    if len(sv_idx) == 0:
        return -y # If no SVs yet, u is just b (0), so E = -y
    
    # 1. Isolate Support Vector data
    X_sv = x[sv_idx]
    alpha_y_sv = alpha[sv_idx] * y[sv_idx]
    
    # 2. Compute Kernel Sub-matrix K(all_points, support_vectors)
    # ||x - x_sv||^2 = ||x||^2 + ||x_sv||^2 - 2 * x @ x_sv.T
    # Shape: (N, len(sv_idx))
    dist_matrix = sq_norm_X.reshape(-1, 1) + sq_norm_X[sv_idx].reshape(1, -1) - 2 * np.dot(x, X_sv.T)
    K_sub = np.exp(-gamma * dist_matrix)
    
    # 3. Compute u = K_sub @ (alpha * y)_sv + b
    u = np.dot(K_sub, alpha_y_sv) + b
    
    # 4. Return E = u - y
    return u - y

def compute_primal_objective(alpha, b, x, y, C, sq_norm_X, gamma = 1):
    # 1. Calculate f(x) for all points (use the kernel trick)
    # This is effectively what your error_cache + y was
    sv_idx = np.where(alpha > 1e-9)[0]
    X_sv = x[sv_idx]
    ay_sv = alpha[sv_idx] * y[sv_idx]
    
    # Kernel between all points and SVs
    dist = sq_norm_X.reshape(-1, 1) + sq_norm_X[sv_idx].reshape(1, -1) - 2 * np.dot(x, X_sv.T)
    K_sub = np.exp(-gamma * dist)
    
    u = np.dot(K_sub, ay_sv) + b
    
    # 2. Compute the norm of the weights in Feature Space
    # ||w||^2 = sum(alpha_i * alpha_j * y_i * y_j * K_ij)
    # Note: This is 2 * (sum(alpha) - Dual_Objective)
    K_sv_sv = np.exp(-gamma * (sq_norm_X[sv_idx].reshape(-1, 1) + sq_norm_X[sv_idx].reshape(1, -1) - 2 * np.dot(X_sv, X_sv.T)))
    weight_norm_sq = ay_sv.T @ K_sv_sv @ ay_sv
    
    # 3. Compute Hinge Loss
    hinge_loss = np.sum(np.maximum(0, 1 - y * u))
    
    primal_obj = 0.5 * weight_norm_sq + C * hinge_loss

    return primal_obj

def SMO(x, y, max_iter, C, track_progress=False):
    N = x.shape[0]
    alpha = np.zeros(N)
    b = 0.0
    sq_norm_X = np.sum(x**2, axis=1)
    error_cache = -y.astype(np.float64)
    history = []
    alphas_list = []
    if track_progress:
        if N <= 5000: 
            obj = compute_dual_objective(alpha, y, x, sq_norm_X)
            history.append(obj)
        alphas_list.append(alpha)
    
    # Heuristic control variables
    entire_set = True
    num_changed_alphas = 0
    current_iter = 0
    old_iter = -1
    while (num_changed_alphas > 0 or entire_set) and (current_iter < max_iter):
        num_changed_alphas = 0
        
        # Decide which indices to check
        if entire_set:
            indices = range(N)
        else:
            # Only points that are "floating" between 0 and C
            indices = np.where((alpha > 0) & (alpha < C))[0]
        for j in indices:
            # Check KKT Violation
            yj, Ej = y[j], error_cache[j]
            if (Ej*yj < -1e-8 and alpha[j] < C) or (Ej*yj > 1e-8 and alpha[j] > 0):
                diff = np.abs(error_cache - Ej)
                top50 = np.argsort(diff)[-50:][::-1]
                #i = np.argmax(np.abs(error_cache - Ej)) 
                for i in top50:
                    yj, Ej = y[j], error_cache[j]
                    Ei = error_cache[i]
                    # KKT violation
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    yi = y[i]

                    if yi == yj:
                        L = L_equal(C, alpha_i_old, alpha_j_old)
                        H = H_equal(C, alpha_i_old, alpha_j_old)
                    else:
                        L = L_not_equal(C, alpha_i_old, alpha_j_old)
                        H = H_not_equal(C, alpha_i_old, alpha_j_old)

                    if L == H:
                        continue

                    eta = get_eta(i, j, x, sq_norm_X)
                    if eta <= 0: 
                        continue
                    
                    alpha_j = np.clip(update_alpha_j(alpha_j_old, Ei, Ej, yj, eta), L, H)
                    
                    delta_j = alpha_j_old - alpha_j
                    
                    if np.abs(delta_j) < 1e-8: # No change
                        continue

                    alpha_i = update_alpha_i(delta_j, alpha_i_old, yi, yj)
                    b_old = b
                    b = update_b(b, Ei, Ej, yi, yj, alpha_i, alpha_j, alpha_i_old, alpha_j_old, x, i, j, C, sq_norm_X)

                    alpha[i], alpha[j] = alpha_i, alpha_j
                    # Compute columns of Kernel matrix for i and j (gamma fixed to 1)
                    ki = np.exp(-1 * (sq_norm_X + sq_norm_X[i] - 2 * np.dot(x, x[i])))
                    kj = np.exp(-1 * (sq_norm_X + sq_norm_X[j] - 2 * np.dot(x, x[j])))
                    
                    # Update the entire cache: E_new = E_old + delta_f + delta_b
                    error_cache += (yi * (alpha_i - alpha_i_old) * ki) + (yj * (alpha_j - alpha_j_old) * kj) + (b - b_old)

                    alpha[i], alpha[j] = alpha_i, alpha_j
                    
                    num_changed_alphas += 1
                    current_iter += 1
                if track_progress:
                    if old_iter < current_iter:
                        if N <= 5000: 
                            obj = compute_dual_objective(alpha, y, x, sq_norm_X)
                            history.append(obj)
                            print(f"Iter {current_iter}: Objective = {obj:.4f}, Alphas Changed = {num_changed_alphas}")
                            old_iter = current_iter
                        assert np.abs(np.sum(alpha * y)) < 1e-5
                        alphas_list.append(alpha.copy())

        if entire_set:
            # After a full pass, switch to non-bound points
            entire_set = False
        elif num_changed_alphas == 0:
            # If no non-bound points changed, check everything again to be sure
            entire_set = True
        
      
        

    return alpha, b, alphas_list, history

if __name__ == "__main__":
    from sklearn.svm import SVC

    Q, y, x =  generate_test_data_2d(1000)
    C = 1
    alpha, b, alphas_list, history = SMO(x, y, 5000, C, track_progress=True)
    #create_svm_gif_kernelized(x, y, alphas_list, history, C = C, kernel = rbf_kernel_matrix, filename= "SMO.gif")




    # Train SKLearn version
    clf = SVC(C=C, kernel='rbf', gamma=1.0)
    clf.fit(x, y)
    print(f"My SV count: {np.sum((alpha > 1e-5) * (alpha < C))}")
    print(f"SKLearn SV count: {len(clf.support_)}")

    # 1. Get predictions from your model
    # We use the final alpha and b to calculate f(x) for all training points
    sv_idx = np.where(alpha > 1e-9)[0]
    # Use your existing SVM_output_k or a vectorized version
    my_preds = np.array([SVM_output_k(i, x, y, alpha, b, sv_idx, np.sum(x**2, axis=1)) for i in range(len(x))])

    # 2. Get predictions from SKLearn
    # decision_function returns the distance to the hyperplane
    sk_preds = clf.decision_function(x)

    # 3. Compare the first 10 values
    print("Index | My f(x) | SKLearn f(x) | Difference")
    print("-" * 45)
    for i in range(10):
        diff = np.abs(my_preds[i] - sk_preds[i])
        print(f"{i:5} | {my_preds[i]:7.4f} | {sk_preds[i]:7.4f} | {diff:9.4f}")

    # 4. Check label agreement
    my_labels = np.sign(my_preds)
    sk_labels = clf.predict(x)
    mismatch = np.sum(my_labels != sk_labels)
    print(f"\nLabel mismatches: {mismatch} out of {len(y)}")