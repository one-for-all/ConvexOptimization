import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [x + y == 1,
               x - y >= 1]

obj = cp.Minimize((x-y)**2)

prob = cp.Problem(obj, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:", x.value, y.value)

print("=============")

prob2 = cp.Problem(cp.Maximize(x+y), prob.constraints)
print("optimal value:", prob2.solve())

print("================")

constraints = [x+y <= 3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("optimal value:", prob3.solve())

