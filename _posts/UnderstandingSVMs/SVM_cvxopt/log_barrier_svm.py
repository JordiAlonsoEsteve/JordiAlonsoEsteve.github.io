import numpy as np
from test_utils import test_derivative
from plotting_utils import create_svm_gif, plot_svm_evolution
from data_utils import generate_test_data, generate_test_data_2d, construct_Q, construct_Q_rbf

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
    while True: # WE ARE MAXIMIZING!
        if f_0(alpha + lr * delta_alpha, Q,  nu, t, y, C) >= f_0(alpha, Q,  nu, t, y, C) + q * lr * (g_0(alpha, Q,  nu, t, y, C).T @ delta_alpha):
            break
        else:
            lr = lr * b
    return lr

def phase_II(x, y, C, alpha, nu,  t = 0.1, epsilon = 1e-5, Q_function = construct_Q):
    alphas_list = []
    value_function = []
    nu_list = []
    assert np.abs(y @ alpha) < 1e-10 and (alpha > 0).all() and (alpha < C).all(), "Values are not feasible!"
    Q = Q_function(x, y)
    while True:
        iter = 1
        while True:
            delta_alpha, delta_nu =_solve_NS_phase_II(alpha, Q, nu, t, y, C)
            lr = backtracking_armijos(alpha, Q, nu, t, y, C, delta_alpha)

            nu = nu + lr * delta_nu
            alpha = alpha + lr * delta_alpha

            g_a = CP_L_grad_alpha(alpha, Q, nu, t, y, C)
            g_nu = CP_L_grad_nu(alpha, Q, nu, t, y, C)
            if  np.linalg.norm(np.concatenate([g_a, [g_nu]]))  < epsilon: # Just check the gradient is close to 0
                # This is the gradient of the of the Lagrangian!
                break
            iter += 1
            nu_list.append(nu)
            alphas_list.append(alpha)
            value_function.append(f_0(alpha, Q, nu, 0, y, C))
            if iter > 50:
                print(f"Inner loop struggling to converge, residual norm: {np.linalg.norm(np.concatenate([g_a, [g_nu]]))}")
                print("Opt. loop continued...")
                break
        if 2*t < epsilon:
            break

        t = t * 0.01
    
    print(f"Phase II completed, t = {t}")            
    
    return alpha, alphas_list, value_function, nu_list


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


def test_phaseII():
        n = 100
        C = 10000
        t = 1
        nu = 1.0
        s = 100.0 # In Phase I, s must be large enough so alpha is feasible (essentially larger than C)
        alpha = np.zeros(shape=(n, ))
        #Q, y, x = generate_test_data(n, p = 2)
        Q, y, x = generate_test_data_2d(int(n/2))
        alpha, _, nu = phase_I(alpha, y, C, s=s, t=t , nu = nu)
        alpha, alphas_list, value_function, nu_list= phase_II(x=x, y=y, C= C, alpha= alpha, nu= nu, Q_function=construct_Q_rbf)
        plot_svm_evolution(x, y, alphas_list, value_function, C = C, filename= "LB.gif")

if __name__ == "__main__":
   
    test_phaseII()