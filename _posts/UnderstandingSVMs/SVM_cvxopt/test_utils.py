import numpy as np

def test_derivative(f, f_prime, inputs, variable_of_interest, epsilon=1e-6, tolerance=1e-4):
    """
    Validates analytical derivatives where f and f_prime take the full dict.
    """
    # 1. Calculate the analytical derivative
    # This now passes the WHOLE dict to your gradient function
    analytical_grad = f_prime(**inputs)
    # 2. Setup for numerical perturbation
    # Grab the current value of the variable we care about
    x_val = np.copy(np.atleast_1d(inputs[variable_of_interest]))
    # Determine output dimension by calling f once
    sample_output = np.atleast_1d(f(**inputs))
    m = sample_output.size
    n = x_val.size
    # numerical_grad will be (m, n)
    # If m=1 (scalar function), it behaves like a normal gradient vector
    numerical_grad = np.zeros((m, n))

    # 3. Finite Difference Loop
    for i in range(x_val.size):
        # Create two deep copies of the dictionary to perturb
        # Use dict(inputs) for a shallow copy, then replace the specific array
        inputs_plus = inputs.copy()
        inputs_minus = inputs.copy()
        
        # Perturb the i-th component of the target variable
        x_plus = np.copy(x_val)
        x_minus = np.copy(x_val)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        # Update the dictionaries with the perturbed values
        # (Handling scalar vs vector)
        inputs_plus[variable_of_interest] = x_plus.reshape(x_val.shape) if x_val.ndim > 0 else x_plus[0]
        inputs_minus[variable_of_interest] = x_minus.reshape(x_val.shape) if x_val.ndim > 0 else x_minus[0]
        
        # Evaluate the function using the full dictionaries
        f_plus = f(**inputs_plus)
        f_minus = f(**inputs_minus)
        numerical_grad[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # Ensure shape consistency for comparison
    numerical_grad = numerical_grad.reshape(analytical_grad.shape)

    # 4. Compare
    error = np.linalg.norm(analytical_grad - numerical_grad) / (np.linalg.norm(analytical_grad) + 1e-10)
        
    return error