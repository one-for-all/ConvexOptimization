import matplotlib.pyplot as plt
import numpy as np


def objective(x):
    return x ** 2 + 1


def lagrangian(x, lambda_):
    return (1 + lambda_) * x ** 2 - 6 * lambda_ * x + 1 + 8 * lambda_


x = np.linspace(-5, 5, 500)
f = objective(x)

# Plot objective vs x
plt.xlabel("x")
plt.ylabel("objective")
plt.title("primal problem: f vs x")
plt.ylim(ymax=max(f))
plt.plot(x, f, label="primal")
plt.fill_between(x, min(f), max(f), where=(x >= 2) & (x <= 4), alpha=0.5, label="feasible set")

# Plot optimal point and value
x_opt = 2
p_opt = objective(x_opt)
plt.plot(x_opt, p_opt, 'ro')

# Plot lagrangian for a few positive lambda
for lambda_ in range(1, 4):
    l = lagrangian(x, lambda_)
    plt.plot(x, l)
    print("lambda: {}, infimum: {}".format(lambda_, min(l)))

# Show the plot
plt.legend()
plt.show()

print("=============================")
# Plot Lagrange dual function
plt.figure()
plt.title("Lagrange dual function: g vs. lambda")
lambda_ = np.linspace(-0.9, 5, 100)
g = -9 * lambda_ ** 2 / (1 + lambda_) + 1 + 8 * lambda_
plt.plot(lambda_, g)
d_opt = max(g)
lambda_opt = lambda_[np.where(g == d_opt)]
print("d optimal: {}, lambda: {}".format(d_opt, lambda_opt))
plt.show()

print("==============================")
# Plot sensitivity function
u1 = np.linspace(-1, 5 / 4, 100)
p1 = (3 - 2 * np.sqrt(1 + u1)) ** 2 + 1
u2 = np.linspace(5 / 4, 3, 100)
p2 = np.ones_like(u2)

u = np.concatenate([u1, u2])
p = np.concatenate([p1, p2])
plt.figure()
plt.title("sensitivity function: p* vs u")
plt.plot(u, p)
plt.show()
