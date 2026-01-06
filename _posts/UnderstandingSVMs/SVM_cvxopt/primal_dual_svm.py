import numpy as np
from data_utils import generate_test_data, generate_test_data_2d, construct_Q, construct_Q_rbf, rbf_kernel_matrix
from plotting_utils import create_svm_gif_linear, create_svm_gif_kernelized
from tqdm import tqdm

def f_0(alpha, Q, lambd, nu, C, y, t):
    return np.sum(alpha) - 0.5 *alpha.T @ Q @ alpha

def r_dual(alpha, Q, lambd, nu, C, y, t):
    # Note that lambda now has the size of 2n!
    n = alpha.size
    pn_one_diag = np.concatenate([-np.eye(n), np.eye(n)]) # 2n rows n columns
    return 1 - Q @ alpha - lambd.T @ pn_one_diag + nu * y

def r_cent(alpha, Q, lambd, nu, C, y, t):
    alphas_concat = np.concatenate([-alpha, alpha - C]) # 2n rows n columns
    return - np.diag(lambd) @ alphas_concat - t

def r_pri(alpha, Q, lambd, nu, C, y, t):
    return y @ alpha

# first row
def g_r_dual_alpha(alpha, Q, lambd, nu, C, y, t):
    return -Q

def g_r_dual_lambd(alpha, Q, lambd, nu, C, y, t):
    n = y.size
    pn_one_diag = np.concatenate([-np.eye(n), np.eye(n)])
    return  - pn_one_diag.T


def g_r_dual_nu(alpha, Q, lambd, nu, C, y, t):
    return y

# second row

def g_r_cent_alpha(alpha, Q, lambd, nu, C, y, t):
    n = y.size
    pn_one_diag = np.concatenate([-np.eye(n), np.eye(n)])
    return - np.diag(lambd) @ pn_one_diag

def g_r_cent_lambd(alpha, Q, lambd, nu, C, y, t):
    return - np.diag(np.concatenate([-alpha, alpha - C]))


# third row
def g_r_prim_dual(alpha, Q, lambd, nu, C, y, t):
    return y.T


def residual_matrix(alpha, Q, lambd, nu, C, y, t):
    r_du = r_dual(alpha, Q, lambd, nu, C, y, t)
    r_ce = r_cent(alpha, Q, lambd, nu, C, y, t)
    r_pr = r_pri(alpha, Q, lambd, nu, C, y, t)
    return np.concatenate([r_du, r_ce, [r_pr]])


def KKT_matrix(alpha, Q, lambd, nu, C, y, t):
    n = alpha.size
    # top row
    top = np.hstack([
        g_r_dual_alpha(alpha, Q, lambd, nu, C, y, t) + 1e-12,   # n x n
        g_r_dual_lambd(alpha, Q, lambd, nu, C, y, t),   # n x 2n
        g_r_dual_nu(alpha, Q, lambd, nu, C, y, t).reshape(-1,1) # n x 1
    ])
    
    # middle row
    middle = np.hstack([
        g_r_cent_alpha(alpha, Q, lambd, nu, C, y, t),   # 2n x n
        g_r_cent_lambd(alpha, Q, lambd, nu, C, y, t),   # 2n x 2n
        np.zeros((2*n, 1))                               # 2n x 1
    ])
    
    # bottom row
    bottom = np.hstack([
        g_r_prim_dual(alpha, Q, lambd, nu, C, y, t).reshape([1, n]),    # 1 x n
        np.zeros((1, 2*n)),                              # 1 x 2n
        np.zeros((1, 1))                                 # 1 x 1
    ])
    
    # stack vertically
    return np.vstack([top, middle, bottom])

def line_search(alpha, Q, lambd, nu, C, y, t, delta_alpha, delta_lambd, delta_nu, a = 0.1, b = 0.3):

    # FEASIBILITY CHECKS:
    # for lambda; find the largest possible s <= 1 that gives all lambda positive
    if np.sum(delta_lambd < 0) > 0:
        ratios = (lambd / delta_lambd)[delta_lambd < 0] 
        s = np.min([1, np.min(-ratios)]) # Not very intuitive but it works. To see this let \nabla lambd be 1 and the ratio below 1. Then is easy to see that
        # using this we get 0.
    else:
        s = 1
    
    s = 0.99*s
    
    alpha_p = alpha + s * delta_alpha
    lambd_p = lambd + s * delta_lambd
    nu_p = nu + s * delta_nu
    

    while np.sum(alpha_p >= 0) != alpha_p.size or np.sum(alpha_p <= C) != alpha_p.size:
        s = b * s
        alpha_p = alpha + s * delta_alpha
        lambd_p = lambd + s * delta_lambd
        nu_p = nu + s * delta_nu

    # FUNCTION DECREASE:
    # to understand the (1-a*s) instead of just 1, think on a parabola. There are certantly directions that minimize the function but not as much as we
    # would like. In that sense, we do have the kind of behaviour were the decrease in the residuals increases when we decrease s until it decreases again. U
    while np.linalg.norm(residual_matrix(alpha_p, Q, lambd_p, nu_p, C, y, t)) > (1 - a*s) * np.linalg.norm(residual_matrix(alpha, Q, lambd, nu, C, y, t)):
        s = b * s
        alpha_p = alpha + s * delta_alpha
        lambd_p = lambd + s * delta_lambd
        nu_p = nu + s * delta_nu
    
    return s


def primal_dual_SVM(x, y, C, mu = 0.1, t = 1, Q_function = construct_Q, epsilon = 1e-5):
    Q = Q_function(x, y)
    n = y.size
    m = 2*n

    alpha = np.zeros(shape=(n, )) + 1e-5
    assert np.sum(alpha < C) == n
    lambd = np.ones((m, ))
    nu = 1

    alphas_list = []
    value_function = []
    nu_list = []

    max_iter = 5000
    eta_hat = np.inf
    pbar = tqdm(range(max_iter), desc="Newton")
    for it in pbar:
        pbar.set_postfix(eta_hat=f"{eta_hat:.3e}")
        alphas_list.append(alpha)
        nu_list.append(nu)
        value_function.append(f_0(alpha, Q, lambd, nu, C, y, t))

        # surrogate duality gap
        alphas_concat = np.concatenate([-alpha, alpha - C])
        eta_hat = -np.sum(lambd * alphas_concat)
        t = mu * eta_hat / m

        r_du = r_dual(alpha, Q, lambd, nu, C, y, t)
        r_ce = r_cent(alpha, Q, lambd, nu, C, y, t)
        r_pr = r_pri(alpha, Q, lambd, nu, C, y, t)

        if np.linalg.norm(r_du) <= epsilon and np.linalg.norm(r_pr) <= epsilon and eta_hat <= epsilon:
            print(f"Converged at iteration {it}")
            break

        r = - np.concatenate([r_du, r_ce, [r_pr]])
        KKT = KKT_matrix(alpha, Q, lambd, nu, C, y, t)
        # --- Newton step ---
        try:
            sol = np.linalg.solve(KKT, r)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(KKT, r, rcond=None)[0]
            break
        
        delta_alpha = sol[:n]
        delta_lambda = sol[n:3*n] # cuz is 2n = 3n-n
        delta_nu = sol[-1]
        
        lr = line_search(alpha, Q, lambd, nu, C, y, t, delta_alpha, delta_lambda, delta_nu)
        # update variables
        alpha =alpha +  lr * delta_alpha
        lambd =lambd +  lr * delta_lambda
        nu = nu + lr * delta_nu
        

    return alpha, lambd, nu, alphas_list, value_function, nu_list

if __name__ == "__main__":
    from test_utils import test_derivative

    
    n = 10
    C = 100
    t = 1
    nu = 1.0
    alpha = np.ones(shape=(n, ))
    Q, y, x = generate_test_data(n, p = 2)
    lambd = np.ones((2*n, )) # two restrictions for each alpha!

    input = {
        'alpha': alpha,
        'y': y,
        'C': C,
        't': t,
        'nu': nu,
        'Q': Q,
        'lambd':lambd
    }

    assert test_derivative(r_dual, g_r_dual_alpha, input, variable_of_interest= "alpha") < 1e-5
    assert test_derivative(r_dual, g_r_dual_lambd, input, variable_of_interest= "lambd") < 1e-5
    assert test_derivative(r_dual, g_r_dual_nu, input, variable_of_interest= "nu") < 1e-5

    assert test_derivative(r_cent, g_r_cent_alpha, input, variable_of_interest= "alpha") < 1e-5
    assert test_derivative(r_cent, g_r_cent_lambd, input, variable_of_interest= "lambd") < 1e-5
    

    n = 100
    C = 1
    t = 1
    nu = 1.0
    #Q, y, x = generate_test_data(n, p = 2)
    Q, y, x = generate_test_data_2d(int(n/2))
    
    alpha, lambd, nu, alphas_list, value_function, nu_list = primal_dual_SVM(x, y, C, Q_function= construct_Q_rbf)
    create_svm_gif_kernelized(x, y, alphas_list, value_function, C = C, kernel = rbf_kernel_matrix, filename= "PD.gif")