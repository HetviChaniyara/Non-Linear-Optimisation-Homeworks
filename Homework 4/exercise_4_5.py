import numpy as np

def newton(grad, hess, x_0):
    tol = 0.001
    nummax = 1000
    x = x_0.astype(float)
    
    for num in range(nummax):
        g = grad(x)
        normgrad = np.linalg.norm(g)
        
        # check if tol met, if yes, return our variables
        if normgrad < tol:
            return x, normgrad, num
        
        h = hess(x)
   
        # same as H^-1*-g but computationally more stable
        delta_x = np.linalg.solve(h, -g)

            
        x = x + delta_x
    g_final = grad(x)
    return x, np.linalg.norm(g_final), nummax