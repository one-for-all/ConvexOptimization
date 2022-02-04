import cvxpy as cp


def problem_a():
    x = cp.Variable(2)
    y = cp.Variable(2)
    constraints = [cp.norm(cp.hstack([x + 2 * y, x - y])) == 0]

    prob = cp.Problem(cp.Minimize(sum(x)), constraints)
    # prob.solve()
    # Equality constraint only takes affine functions on both sides.
    # The norm function is convex.
    print("status: ", prob.status)
    print("==========================")

    constraints = [x + 2 * y == 0, x - y == 0]
    prob = cp.Problem(cp.Minimize(sum(x)), constraints)
    prob.solve()
    print("status: ", prob.status)


def problem_b():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [cp.square(cp.square(x + y)) <= x - y]

    prob = cp.Problem(cp.Minimize(sum(x)), constraints)
    prob.solve()
    # It passed the check here.
    # But according to the DCP rule-set, it shouldn't,
    # because square() is convex, but neither increasing nor decreasing.
    # For it to be considered convex, the argument needs to be affine.
    print("status: ", prob.status)
    print("==========================")

    constraints = [(x + y) ** 4 <= x - y]
    prob = cp.Problem(cp.Minimize(sum(x)), constraints)
    prob.solve()
    print("status: ", prob.status)


def problem_c():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [1 / x + 1 / y <= 1,
                   x >= 0,
                   y >= 0]

    prob = cp.Problem(cp.Minimize(x), constraints)
    # prob.solve()
    # 1/x is not a convex function over its entire domain.
    print("status: ", prob.status)
    print("==========================")

    constraints = [cp.inv_pos(x) + cp.inv_pos(y) <= 1]
    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("optimal value:", prob.value)
    print("optimal var:", x.value)


def problem_d():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [cp.norm(cp.hstack([cp.maximum(x, 1),
                                      cp.maximum(y, 2)])) <= 3 * x + y]

    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    # It works here, but should not according to DCP rule-set.
    # norm() is a convex function, but neither increasing nor decreasing,
    # so should only take affine arguments.
    print("status: ", prob.status)
    print("optimal value:", prob.value)
    print("==========================")

    t = cp.Variable(2)
    constraints = [cp.norm(cp.hstack([t[0], t[1]])) <= 3 * x + y,
                   cp.maximum(x, 1) <= t[0],
                   cp.maximum(y, 2) <= t[1]]
    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("optimal value:", prob.value)


def problem_e():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [x * y >= 1,
                   x >= 0,
                   y >= 0]

    prob = cp.Problem(cp.Minimize(x), constraints)
    # prob.solve()
    # x*y is not a convex function
    print("status: ", prob.status)
    print("==========================")

    constraints = [x >= cp.inv_pos(y),
                   x >= 0]
    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)


def problem_f():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [(x + y) ** 2 / cp.sqrt(y) <= x - y + 5]

    prob = cp.Problem(cp.Minimize(x), constraints)
    # Cannot have division with expression in DCP rule-set.
    # Can use the quad_over_lin(a, b) function, which is a*a/b,
    # for a in real number, and b in positive real number.
    # The first argument is neither increasing nor decreasing,
    # so can take affine argument x+y.
    # The second argument is decreasing, so can take concave
    # function, i.e sqrt(y).
    print("status: ", prob.status)
    print("==========================")

    constraints = [cp.quad_over_lin(x + y, cp.sqrt(y)) <= x - y + 5]
    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)


def problem_g():
    x = cp.Variable()
    y = cp.Variable()
    constraints = [x ** 3 + y ** 3 <= 1,
                   x >= 0,
                   y >= 0]

    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    # It passes the check here, but does not meets the DCP rule-set.
    # x**3 + y**3 is not convex over entire domain
    print("status: ", prob.status)
    print("==========================")

    constraints = [cp.power(x, 3) + cp.power(y, 3) <= 1]
    # power function for odd power, is inf for x < 0.

    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("optimal value:", prob.value)
    print("optimal var:", x.value)


def problem_h():
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    constraints = [x + z <= 1 + cp.sqrt(x * y - z ** 2),
                   x >= 0,
                   y >= 0]

    prob = cp.Problem(cp.Minimize(x), constraints)
    # prob.solve()
    # xy is not concave
    print("status: ", prob.status)
    print("==========================")

    constraints = [x + z <= 1 + cp.geo_mean(cp.hstack([y, x - cp.quad_over_lin(z, y)])),
                   x >= 0,
                   y >= 0]

    prob = cp.Problem(cp.Minimize(x), constraints)
    prob.solve()
    print("status: ", prob.status)
    print("optimal value:", prob.value)
    print("optimal var:", x.value)


problem_h()
