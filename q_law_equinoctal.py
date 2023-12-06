import jax.numpy as jnp
import jax

# equinoctal q-law: https://indico.esa.int/event/111/contributions/346/attachments/336/377/Final_Paper_ICATT.pdf


@jax.jit
def get_oe_dot_xx(oe, F, mu):
    """
    Maximum rate of change of equinoctal Q-law variables
    oe: orbital elements (equinoctal) (a, f, g, h, k, L)
    F: control parameter (Can just use F = 1)
    mu: gravitational parameter (Can just use mu = 1)
    """
    a, f, g, h, k, L = oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]
    e = jnp.sqrt(f**2 + g**2)
    p = a * (1 - e**2)
    a_dot_xx = 2 * F * a * jnp.sqrt(a / mu) * jnp.sqrt(
        (1 + jnp.sqrt(f**2 + g**2)) / (1 - jnp.sqrt(f**2 + g**2)))
    f_dot_xx = 2 * F * jnp.sqrt(p / mu)
    g_dot_xx = f_dot_xx
    s_squared = 1 + h**2 + k**2
    h_dot_xx = 0.5 * F * jnp.sqrt(
        p / mu) * s_squared / (jnp.sqrt(1 - g**2) + f)
    k_dot_xx = 0.5 * F * jnp.sqrt(
        p / mu) * s_squared / (jnp.sqrt(1 - f**2) + g)
    return jnp.array([
        a_dot_xx,
        f_dot_xx,
        g_dot_xx,
        h_dot_xx,
        k_dot_xx,
        1,
    ])


@jax.jit
def get_d(oe, oe_T):
    """
    Difference between current and target orbital elements
    oe: orbital elements (equinoctal) (a, f, g, h, k, L)
    oe_T: target orbital elements (equinoctal) (a, f, g, h, k, L)
    """
    # print(oe, oe_T)
    d = oe - oe_T
    # set the last element to 0 as we don't care about L
    d = d.at[-1].set(0)
    return d


@jax.jit
def Q_value(oe, oe_T):
    """
    Q-value for equinoctal Q-law
    oe: orbital elements (equinoctal) (a, f, g, h, k, L)
    oe_T: target orbital elements (equinoctal) (a, f, g, h, k, L)
    """
    F, mu = 1, 1
    oe_dot_xx = get_oe_dot_xx(oe, F, mu)
    return jnp.sum((get_d(oe, oe_T) / oe_dot_xx)**2)
