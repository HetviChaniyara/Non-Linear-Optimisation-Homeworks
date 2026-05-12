import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess
from exercise_4_5 import newton 

def run_test():
    x0 = np.array([-1.2, 1.0])

    x_min, final_norm, iterations = newton(rosen_der, rosen_hess, x0)
    
    print(f"Approximate Solution: {x_min}")
    print(f"Gradient Norm: {final_norm:.6f}")
    print(f"Iterations: {iterations}")

    # checking against true values
    true_solution = np.ones_like(x0)
    is_correct_pos = np.allclose(x_min, true_solution, atol=1e-3)
    is_correct_grad = final_norm < 0.001 

    if is_correct_pos and is_correct_grad:
        print("The algorithm successfully reached the global min")
    else:
        print("\ntest failed")

if __name__ == "__main__":
    run_test()