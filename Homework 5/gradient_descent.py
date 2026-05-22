import numpy as np
from scipy.optimize import rosen, rosen_der

def gradient_method(func, gradient, x_0, delta=0.1, max_iter=1000, tol=1e-5):
    x = np.array(x_0, dtype=float)
    num = 0
    
    for k in range(max_iter):
        grad = np.array(gradient(x), dtype=float)
        normgrad = np.linalg.norm(grad)
        
        # stop if converged
        if normgrad < tol:
            return x, normgrad, num
    
        d = -grad
        
        # sigma_0
        sigma = 1.0  
        
        # compute armijo line factor
        armijo_factor = delta * np.dot(grad, d)
        
        # Reduce stepsize by half until function below armijo line
        while func(x + sigma * d) > func(x) + sigma * armijo_factor:
            sigma /= 2.0  
            
        # update x_k+1
        x = x + sigma * d
        num += 1

    grad = np.array(gradient(x), dtype=float)
    normgrad = np.linalg.norm(grad)
    return x, normgrad, num

if __name__ == "__main__":
    x_0 = np.array([-1.2, 1.0])
    
    print("Gradient Descent:")
    print(f"x_0: {x_0}\n")
    
    x_opt, normgrad, num_iterations = gradient_method(rosen, rosen_der, x_0, delta=0.1)

    print(f"x*: {x_opt}")
    print(f"x global min: [1,1]")
    print(f"f at x*: {rosen(x_opt)}")
    print(f"Norm of the gradient: {normgrad}")
    print(f"Num of iterations: {num_iterations}")
