import cvxpy as cp
import numpy as np

N = 4
M = 5
x = cp.Variable(N)
A = np.array([[1, 2, 0, 1],
              [0, 0, 3, 1],
              [0, 3, 1, 1],
              [2, 1, 2, 5],
              [1, 0, 3, 2]])
c_max = np.array([100] * M)
constraints = [x >= 0,
               A @ x <= c_max]

p = np.array([3, 2, 7, 6])
p_disc = np.array([2, 1, 4, 2])
q = np.array([4, 10, 5, 10])

# Build up the objective function
rs = []
for i in range(N):
    r = cp.minimum(p[i] * x[i],
                   p[i] * q[i] + p_disc[i] * (x[i] - q[i]))
    rs.append(r)

prob = cp.Problem(cp.Maximize(sum(rs)), constraints)
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:", x.value)
print("===============================")

# =======================
# Create the LP problem in standard form
t = cp.Variable(N)
constraints = [-x <= 0,
               A @ x <= c_max]

# Add transformed additional constraints
for i in range(N):
    constraints.append(-(p[i] * x[i]) <= t[i])
    constraints.append(-(p[i] * q[i] + p_disc[i] * (x[i] - q[i])) <= t[i])

prob = cp.Problem(cp.Minimize(sum(t)), constraints)
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:")
print("  t: ", t.value)
print("  x: ", x.value)

print("#### analysis: ")
print("optimal activity levels: ", x.value)
print("revenue generated: ", -t.value)
print("total revenue: ", -prob.value)
print("average price per unit: ", -t.value / x.value)
