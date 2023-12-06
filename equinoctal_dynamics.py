import dis
import jax.numpy as jnp
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from functools import partial

import numpy as np
from tqdm.auto import tqdm
import cvxpy as cvx

from q_law_equinoctal import Q_value

# equinoctal orbital elements: https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf

# equinoctal state derivative: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9438362


def keplerian_to_equinoctal(oe):
    """Convert Keplerian orbital elements to equinoctal coordinates."""
    # oe: orbital elements in the order [a, e, i, Omega, omega, theta]
    a, e, i, Omega, omega, theta = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    f = e * jnp.sin(omega + Omega)
    g = e * jnp.cos(omega + Omega)
    h = jnp.tan(i / 2) * jnp.cos(Omega)
    k = jnp.tan(i / 2) * jnp.sin(Omega)
    L = omega + Omega + theta
    return jnp.array([a, f, g, h, k, L])


def equinoctal_to_keplerian(oe):
    """Convert equinoctal coordinates to Keplerian orbital elements."""
    # oe: orbital elements in the order [a, f, g, h, k, L]
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    e = jnp.sqrt(f**2 + g**2)
    i = jnp.arctan2(2 * jnp.sqrt(h**2 + k**2), 1 - h**2 - k**2)
    Omega = jnp.arctan2(k, h)
    omega = jnp.arctan2(g * k - f * k, f * h + g * k)
    theta = L - omega - Omega
    return jnp.array([a, e, i, Omega, omega, theta])


def get_l(oe):
    # semilatus rectum
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    return a * (1 - f**2 - g**2)


def get_r(oe):
    # radius
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    l = get_l(oe)
    return l / (1 + f * jnp.sin(L) + g * jnp.cos(L))


def a_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    l = get_l(oe)
    r = get_r(oe)
    coeff = 2 * a**2 / h
    vec = jnp.array([
        g * jnp.sin(L) - f * jnp.cos(L),
        l / r,
        0,
    ])
    return coeff * jnp.dot(vec, F)


def f_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    l = get_l(oe)
    r = get_r(oe)

    coeff = r / h
    vec = jnp.array([
        -l / r * jnp.cos(L),
        f + (1 + l / r) * jnp.sin(L),
        -g * (h * jnp.cos(L) - k * jnp.sin(L)),
    ])
    return coeff * jnp.dot(vec, F)


def g_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    l = get_l(oe)
    r = get_r(oe)

    coeff = r / h
    vec = jnp.array([
        l / r * jnp.sin(L),
        g + (1 + l / r) * jnp.cos(L),
        f * (h * jnp.cos(L) - k * jnp.sin(L)),
    ])
    return coeff * jnp.dot(vec, F)


def h_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    r = get_r(oe)

    coeff = r / (2 * h)
    vec = jnp.array([
        0,
        0,
        (1 + h**2 + k**2) * jnp.sin(L),
    ])
    return coeff * jnp.dot(vec, F)


def k_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    r = get_r(oe)

    coeff = r / (2 * h)
    vec = jnp.array([
        0,
        0,
        (1 + h**2 + k**2) * jnp.cos(L),
    ])
    return coeff * jnp.dot(vec, F)


def L_dot(oe, F):
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    l = get_l(oe)
    r = get_r(oe)

    coeff = r / h
    vec = jnp.array([
        0,
        0,
        k * jnp.sin(L) - h * jnp.cos(L),
    ])
    return h / r**2 + coeff * jnp.dot(vec, F)


def equinoctal_elements_dot(oe, F):
    # F_comb is the sum of F[:3] and F[3:]
    F_comb = F[:3] + F[3:]
    return jnp.array([
        a_dot(oe, F_comb),
        f_dot(oe, F_comb),
        g_dot(oe, F_comb),
        h_dot(oe, F_comb),
        k_dot(oe, F_comb),
        L_dot(oe, F_comb),
    ])


######################################### BELOW HERE SHOULD MOVE TO A NEW FILE #########################################
#### SCP ####


@partial(jax.jit, static_argnums=(0, ))
@partial(jax.vmap, in_axes=(None, 0, 0))
def linearize(fd: callable, s: jnp.ndarray, u: jnp.ndarray):
    """Linearize the function `fd(s,u)` around `(s,u)`."""
    A, B = jax.jacobian(fd, (0, 1))(s, u)
    c = fd(s, u) - A @ s - B @ u

    return A, B, c


@partial(jax.jit, static_argnums=(0, ))
@partial(jax.vmap, in_axes=(None, 0, None))
def linearize_Q_value(Q_value_func: callable, s: jnp.ndarray,
                      s_goal: jnp.ndarray):
    """Linearize the function `fd(s,u)` around `(s,u)`."""
    A = jax.jacfwd(Q_value_func, 0)(s, s_goal)
    c = Q_value_func(s, s_goal) - A @ s

    return A, c


def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


def solve_scp(fd: callable, Q_value_func: callable, P: np.ndarray,
              Q: np.ndarray, R: np.ndarray, N: int, s_goal: np.ndarray,
              s0: np.ndarray, ru_cont: float, ru_imp: float, ρ: float,
              ρ_ru: float, tol: float, max_iters: int):
    """Solve the problem via SCP."""
    n = s0.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize nominal trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = fd(s_bar[k], u_bar[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        # print('SCP iteration {}'.format(i))
        # print(f"u_bar: {u_bar}")
        print(f"s_bar[-1]: {s_bar[-1]}")
        # print(f"s_goal: {s_goal}")
        s, u, obj = scp_iteration(fd, Q_value_func, P, Q, R, N, s_bar, u_bar,
                                  s_goal, s0, ru_cont, ru_imp, ρ, ρ_ru)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({
            'objective': obj,
            'objective change': diff_obj,
        })

        # print(f"s[-1]: {s[-1]}")

        if diff_obj < tol:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


def scp_iteration(fd: callable, Q_value_func: callable, P: float, Q: float,
                  R: np.ndarray, N: int, s_bar: np.ndarray, u_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray, ru_cont: float,
                  ru_imp: float, ρ: float, ρ_ru: float):
    # fd: the discretized dynamics
    # Q_value_func: the Q value function
    # P, Q, R: the cost matrices/coefficients P (terminal), Q (state), R (control)
    # N: the number of time steps
    # s_bar: the nominal state trajectory
    # u_bar: the nominal control trajectory
    # s_goal: the goal state
    # s0: the initial state
    # ru: the control effort bound
    # ρ: the trust region parameter
    """Solve a single SCP sub-problem."""

    # linearize the dynamics around the nominal trajectory
    A, B, c = linearize(fd, s_bar[:-1], u_bar)

    # linearize the Q value function around the nominal trajectory
    A_Q_value, c_Q_value = linearize_Q_value(
        Q_value_func,
        s_bar,
        s_goal,
    )

    # convert to numpy arrays
    A, B, c = np.array(A), np.array(B), np.array(c)
    A_Q_value, c_Q_value = np.array(A_Q_value), np.array(c_Q_value)
    # we use the above approximate the Q function with a quadratic function, as the original is not convex

    n = 6
    m = R.shape[0]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # Construct and solve the convex sub-problem for SCP.

    # TODO: cost should be the Q function in equinoctal coordinates
    cost_terms = [cvx.quad_form(u_cvx[k], R) for k in range(N)]
    # add in the terms using the approximate Q function
    # running cost (not using this right now):
    # cost_terms += [
    #     (cvx.sum(cvx.multiply(s_cvx, A_Q_value)) + cvx.sum(c_Q_value)) * Q
    # ]
    # terminal cost:
    cost_terms += [
        (cvx.sum(cvx.multiply(s_cvx[N], A_Q_value[N])) + c_Q_value[N]) * P
    ]

    # Add in cost terms as soft constraints on rho if we want
    # constaint_cost = 3e6
    # cost_terms += [
    #     constaint_cost *
    #     cvx.square(cvx.pos(cvx.max(cvx.abs(s_cvx - s_bar)) - ρ))
    # ]
    # cost_terms += [
    #     constaint_cost *
    #     cvx.square(cvx.pos(cvx.max(cvx.abs(u_cvx - u_bar)) - ρ))
    # ]
    # cost_terms += [
    #     constaint_cost *
    #     cvx.square(cvx.pos(cvx.max(cvx.abs(u_cvx[:, :3])) - ru_cont))
    # ]
    # cost_terms += [
    #     constaint_cost *
    #     cvx.square(cvx.pos(cvx.max(cvx.abs(u_cvx[:, 3:])) - ru_imp))
    # ]

    objective = sum(cost_terms)

    constraints = [
        s_cvx[k + 1] == A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k]
        for k in range(N)
    ]
    constraints += [s_cvx[0] == s0]

    # It doesn't like cvx.abs for some reason so we'll go with min and max constraints
    constraints += [cvx.max(s_cvx - s_bar) <= ρ]
    constraints += [cvx.min(s_cvx - s_bar) >= -ρ]

    constraints += [cvx.max(u_cvx - u_bar) <= ρ_ru]
    constraints += [cvx.min(u_cvx - u_bar) >= -ρ_ru]

    constraints += [cvx.max(u_cvx[:, :3]) <= ru_cont]
    constraints += [cvx.min(u_cvx[:, :3]) >= -ru_cont]
    constraints += [cvx.max(u_cvx[:, 3:]) <= ru_imp]
    constraints += [cvx.min(u_cvx[:, 3:]) >= -ru_imp]

    # ############################# END PART (c) ##############################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    # print(prob)
    prob.solve(solver=cvx.ECOS, verbose=False)

    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    if prob.status == 'optimal_inaccurate':
        print('SCP solve warning. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value
    obj = prob.objective.value

    return s, u, obj


# simulation parameters
n = 6  # state dimension
m = 6  # control dimension
s_goal = keplerian_to_equinoctal(jnp.array([1.1, 0.0, 0.0, 0.0, 0.0,
                                            0.0]))  # desired state
s0 = keplerian_to_equinoctal(jnp.array([1, 0.5, 0.5, 0.5, 0.5,
                                        0.0]))  # initial state
dt = 0.25  # discrete time resolution
T = 15.0  # total simulation time

# Dynamics
fd = jax.jit(discretize(equinoctal_elements_dot, dt))

# # SCP parameters
P = 5e2  # terminal state cost matrix
Q = 1e-2  # state cost matrix (CURRENTLY UNUSED)
R_cont = 1e-2 * np.eye(m // 2)  # control cost matrix: continuous thrust
R_imp = 1 * np.eye(m // 2)  # control cost matrix: impulsive thrust
R = np.block([[R_cont, np.zeros((m // 2, m // 2))],
              [np.zeros((m // 2, m // 2)), R_imp]])

ρ = 0.0025  # trust region parameter
ρ_ru = 1e-1  # trust region parameter for control effort
ru_cont = 1e-4  # control effort bound: continuous thrust
ru_imp = 1e-2  # control effort bound: impulsive thrust

tol = 1e-2  # convergence tolerance
max_iters = 500  # maximum number of SCP iterations

# # Solve the problem with SCP
t = np.arange(0., T + dt, dt)
N = t.size - 1
s, u = solve_scp(fd, Q_value, P, Q, R, N, s_goal, s0, ru_cont, ru_imp, ρ, ρ_ru,
                 tol, max_iters)

# plot the results
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()
labels_equinoctal = ['a', 'f', 'g', 'h', 'k', 'L']

labels_keplerian = ['a', 'e', 'i', 'Omega', 'omega', 'theta']
s_keplerian = np.empty_like(s)
for i in range(s.shape[0]):
    s_keplerian[i, :] = equinoctal_to_keplerian(s[i, :])

for i in range(n):
    ax[i].plot(t, s[:, i], label='SCP')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel(labels_equinoctal[i])
    ax[i].legend()
plt.tight_layout()

# plot the controls
fig, ax = plt.subplots(1, m, figsize=(15, 5))
ax = ax.flatten()
labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
for i in range(m):
    ax[i].plot(t[:-1], u[:, i], label='SCP')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel(labels[i])
    ax[i].legend()
plt.tight_layout()

plt.show()

# test by running a simulation with no control
# s = np.zeros((N + 1, n))
# s[0] = s0
# F = 1e-4 * np.array([1, 1, 1])
# for k in range(N):
#     s[k + 1] = fd(s[k], F)

# # Plot the results
# fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# ax = ax.flatten()
# labels = ['a', 'f', 'g', 'h', 'k', 'L']
# for i in range(n):
#     ax[i].plot(t, s[:, i], label='SCP')
#     ax[i].set_xlabel('Time (s)')
#     ax[i].set_ylabel(labels[i])
#     ax[i].legend()
# plt.tight_layout()

# plt.show()
