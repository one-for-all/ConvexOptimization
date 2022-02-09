import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

A = np.loadtxt("illum_data.csv", delimiter=",")
M, N = A.shape
print("{} patches and {} lamps.".format(M, N))

P_MAX = 1
I_TARGET = 1


def compute_objective(p):
    return np.max(np.abs(np.log(np.matmul(A, p))))


def equal_lamp_power():
    gamma_values = np.linspace(0.01, P_MAX, num=200)
    f_values = []
    for gamma in gamma_values:
        p = np.array([gamma] * N).reshape((-1, 1))
        f = np.max(np.abs(np.log(np.matmul(A, p))))
        f_values.append(f)

    plt.plot(gamma_values, f_values)
    # plt.show()

    # Graphically determined that optimal gamma is ~0.345.
    gamma = 0.345
    p = np.array([gamma] * N).reshape((-1, 1))
    f = compute_objective(p)
    print("optimal gamma: ", gamma)
    print("optimal objective value: ", f)


def least_squares_with_saturation():
    b = np.array([I_TARGET] * M)
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    for i in range(len(p)):
        if p[i] > P_MAX:
            p[i] = P_MAX
        elif p[i] < 0:
            p[i] = 0

    f = compute_objective(p)
    print("p: ", p)
    print("f: ", f)


def regularized_least_squares():
    b1 = np.array([I_TARGET] * M)
    found = False
    for ro in np.linspace(0, 5, num=1000):
        b2 = np.array([ro * 0.5] * N)
        b = np.concatenate((b1, b2))
        I = ro * np.identity(N)
        A_prime = np.concatenate((A, I), axis=0)
        p = np.linalg.lstsq(A_prime, b, rcond=None)[0]

        all_within = True
        for p_prime in p:
            if p_prime < 0 or p_prime > 1:
                all_within = False
                break
        if all_within:
            found = True
            break

    f = compute_objective(p)
    print("found solution: ", found)
    print("p: ", p)
    print("f: ", f)


def chebyshev():
    p = cp.Variable(N)
    target = np.array([I_TARGET] * M)
    objective = cp.norm_inf(cp.matmul(A, p) - target)
    constraints = [p >= 0,
                   p <= P_MAX]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("p: ", p.value)
    print("f: ", compute_objective(p.value))


def exact():
    p = cp.Variable(N)
    t1 = cp.matmul(A, p)
    t2 = cp.inv_pos(t1)
    t = cp.hstack([t1, t2])
    objective = cp.maximum(*t)
    constraints = [p >= 0,
                   p <= P_MAX]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("p: ", p.value)
    print("f: ", compute_objective(p.value))


exact()
