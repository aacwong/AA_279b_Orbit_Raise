import jax
import jax.numpy as jnp


@jax.jit
def get_p(oe):
    # semi-latus rectum
    a = oe[0]
    e = oe[1]
    return a * (1 - e**2)


@jax.jit
def get_h(oe, mu):
    # angular momentum
    a = oe[0]
    e = oe[1]
    return jnp.sqrt(mu * a * (1 - e**2))


@jax.jit
def get_r(oe, theta):
    # radius
    p = get_p(oe)
    e = oe[1]
    return p / (1 + e * jnp.cos(theta))


def plot_orbit(ax, oe, **kwargs):
    a, e, i, Omega, omega = oe[0], oe[1], oe[2], oe[3], oe[4]
    theta = jnp.linspace(0, 2 * jnp.pi, 1000)
    r = get_r(oe, theta)
    ax.plot(r * jnp.cos(theta + omega), r * jnp.sin(theta + omega), **kwargs)
