import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der

def backtrace(x, d, f, df_x, t0=1., max_backtracking_iter=100, delta=1e-4, beta=.5):
    """
    Armijo Line Search Backtracking Strategy
    """
    t = t0

    df_x = df_x.reshape(-1, 1)
    d = d.reshape(-1, 1)
    
    for _ in range(max_backtracking_iter):
        lhs = f((x + t * d).flatten())
        armijo = f(x.flatten()) + delta * t * np.dot(df_x.T, d).item()
        
        # first time function is below the armijo line
        if lhs <= armijo:
            break
        # reduce t by factor beta 
        t *= beta
    return t

def BFGS(f, df, x0, eps=1e-8, Ainv0=None, max_iter=100):
    """
    BFGS Optimization routine using the Sherman-Morrison Update.
    """
    n = x0.shape[0]
    if Ainv0 is None:
        Ainv0 = np.eye(n)
    Ainv = np.copy(Ainv0)
    x = np.reshape(x0, (-1, 1))

    x_values = [x.flatten()]

    for i in range(max_iter):
        grad = df(x.flatten()).reshape(-1, 1)
        
        # convergence criteria check
        if np.linalg.norm(grad) < eps:
            print(f"Converged in {i} iterations.")
            break
            
        # descent direction: d = - Ainv * grad
        d = - np.dot(Ainv, grad)
        
        # finding efficient step size
        t = backtrace(x, d, f, grad)
        
        # update x_k+1 and grad(x_k+1)
        x_next = x + t * d
        grad_next = df(x_next.flatten()).reshape(-1, 1)
        
        # update s_k and y_k
        s = x_next - x
        y = grad_next - grad
        
        # for plot
        x_values.append(x_next.flatten())
        x = x_next

        # compute BFGS Inverse update
        rho_inv = np.dot(y.T, s).item()
        rho = 1.0 / rho_inv
        I_mat = np.eye(n)
        Ainv = np.dot((I_mat - rho * np.dot(s, y.T)), np.dot(Ainv, (I_mat - rho * np.dot(y, s.T)))) + rho * np.dot(s, s.T)

    x_values = np.array(x_values)
    
    # Visualization plot generation
    plt.figure(figsize=(8, 6))
    plt.plot(x_values[:, 0], x_values[:, 1], 'o-', label='BFGS Path')
    plt.plot(1, 1, 'r*', markersize=15, label='Global Minimum (1,1)')
    plt.title('BFGS Trajectory on Rosenbrock Function')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return x

# Execute Algo
x = BFGS(rosen, rosen_der, np.array([2.0, 2.0]))
print("Final computed minimum location:\n", x.flatten())
print("Normalized error distance to true minimum:", np.linalg.norm(x.flatten() - np.array([1, 1])) / np.sqrt(2))