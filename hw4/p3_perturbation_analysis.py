import cvxpy as cp
import numpy as np


def perturbed_problem(delta1, delta2):
    u1 = -2 + delta1
    u2 = -3 + delta2

    x = cp.Variable(2)
    A = np.array([[1, -1 / 2],
                  [-1 / 2, 2]])

    objective = cp.Minimize(cp.quad_form(x, A) - x[0])

    def constraint1(x):
        return x[0] + 2 * x[1] - u1

    def constraint2(x):
        return x[0] - 4 * x[1] - u2

    def constraint3(x):
        return 5 * x[0] + 76 * x[1] - 1

    constraints = [constraint1(x) <= 0,
                   constraint2(x) <= 0,
                   constraint3(x) <= 0]

    prob = cp.Problem(objective, constraints)

    prob.solve()

    lambda_ = [constraint.dual_value for constraint in constraints]
    f = [constraint1(x.value), constraint2(x.value), constraint3(x.value)]
    return prob, constraints, x.value, lambda_, f


prob, constraints, x, lambda_, f = perturbed_problem(0, 0)
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:", x)

print("optimal dual var:", lambda_)

print("=======================")
print("verify KKT conditions")
print("f1: {} <= 0".format(f[0]))
print("f2: {} <= 0".format(f[1]))
print("f3: {} <= 0".format(f[2]))
print("lambda1: {} >= 0".format(lambda_[0]))
print("lambda2: {} >= 0".format(lambda_[1]))
print("lambda3: {} >= 0".format(lambda_[2]))
print("lambda1 * f1: {} == 0".format(lambda_[0] * f[0]))
print("lambda2 * f2: {} == 0".format(lambda_[1] * f[1]))
print("lambda3 * f3: {} == 0".format(lambda_[2] * f[2]))


def gradient(x, lambda_):
    return np.array([2 * x[0] - x[1] - 1 + lambda_[0] + lambda_[1] + 5 * lambda_[2],
                     4 * x[1] - x[0] + 2 * lambda_[0] - 4 * lambda_[1] + 76 * lambda_[2]])


print("gradient: {} == 0".format(gradient(x, lambda_)))

print("============================")
print("question(b): perturbed problems' solutions")

prob, _, _, lambda_, _ = perturbed_problem(0, 0)
p_opt_orig = prob.value


def pred_opt_val(delta1, delta2):
    return p_opt_orig - lambda_[0] * delta1 - lambda_[1] * delta2


print("delta1, delta2, p*_pred, p*_exact")
print("0, 0, {}, {}".format(pred_opt_val(0, 0), p_opt_orig))
deltas = [[0, -0.1],
          [0, 0.1],
          [-0.1, 0],
          [-0.1, -0.1],
          [-0.1, 0.1],
          [0.1, 0],
          [0.1, -0.1],
          [0.1, 0.1]]
for delta1, delta2 in deltas:
    p_pred = pred_opt_val(delta1, delta2)
    p_opt = perturbed_problem(delta1, delta2)[0].value
    print("{}, {}, {}, {}".format(delta1, delta2, p_pred, p_opt))
    assert p_pred <= p_opt
