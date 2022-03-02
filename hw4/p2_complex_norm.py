import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

M = 30
N = 100


def complex_norm_problem(norm_func):
    x = cp.Variable(N, complex=True)

    A = np.random.rand(M, N) + np.random.rand(M, N) * 1j
    b = np.random.rand(M) + np.random.rand(M) * 1j
    constraints = [A @ x == b]

    obj = cp.Minimize(norm_func(x))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    print("status:", prob.status)
    print("=============")

    return x


print("complex 2 norm problem")
x_2norm = complex_norm_problem(cp.norm2)
plt.scatter(x_2norm.value.real, x_2norm.value.imag, label="2 norm", color="red")

print("complex inf norm problem")
x_inf_norm = complex_norm_problem(cp.norm_inf)
plt.scatter(x_inf_norm.value.real, x_inf_norm.value.imag, label="inf norm", color="blue")

plt.gca().set_aspect("equal")
plt.legend()
plt.show()
