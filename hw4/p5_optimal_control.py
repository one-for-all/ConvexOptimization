import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

N = 30  # number of time steps
K = 3  # state dimension
A = np.array([[-1, 0.4, 0.8],
              [1, 0, 0],
              [0, 1, 0]])
b = np.array([1, 0, 0.3])
x_des = np.array([7, 2, -6])

x = cp.Variable((N + 1, K))
u = cp.Variable(N)
s = cp.Variable(N)

objective = cp.Minimize(sum(s))

constraints = [u <= s,
               -u <= s,
               2 * u - 1 <= s,
               -2 * u - 1 <= s,
               x[0] == np.zeros(3),
               x[N] == x_des] + [A @ x[i] + b * u[i] == x[i + 1] for i in range(N)]

prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal u:", u.value)

plt.stairs(u.value)
plt.xlabel("time step (t)")
plt.ylabel("control input (u)")
plt.show()

print("=========================")
print("solve directly without transforming to LP")
