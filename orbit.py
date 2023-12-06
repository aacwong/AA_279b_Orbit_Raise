from dataclasses import dataclass

from jax.typing import ArrayLike


@dataclass
class Orbit:
    # orbital elements in the order a, e, i, Omega, omega
    orbital_elements: ArrayLike
    eci_coords: ArrayLike
    M_0: float = 0
    epoch: float = 0
    mu: float = 398600.4418

    @classmethod
    def from_orbital_elements(cls,
                              orbital_elements,
                              M_0=0,
                              epoch=0,
                              mu=398600.4418):
        n = jnp.sqrt(mu / orbital_elements[0]**3)
        theta = theta_0 + n * epoch
        eci_coords = oe_to_eci(orbital_elements, theta, cls.mu)
        return cls(
            jnp.array(orbital_elements),
            jnp.array(eci_coords),
            theta_0=theta_0,
            epoch=epoch,
            mu=mu,
        )

    @classmethod
    def from_eci(cls, eci_coords, M_0=0, epoch=0, mu=398600.4418):
        orbital_elements = eci_to_oe(eci_coords, cls.mu)
        return cls(
            jnp.array(orbital_elements),
            jnp.array(eci_coords),
            theta_0=theta_0,
            epoch=epoch,
            mu=mu,
        )

    @property
    def a(self):
        # semimajor axis
        return self.orbital_elements[0]

    @property
    def e(self):
        # eccentricity
        return self.orbital_elements[1]

    @property
    def i(self):
        # inclination
        return self.orbital_elements[2]

    @property
    def Omega(self):
        # right ascension of the ascending node
        return self.orbital_elements[3]

    @property
    def omega(self):
        # argument of periapsis
        return self.orbital_elements[4]

    @property
    def n(self):
        # mean motion
        return jnp.sqrt(self.mu / self.a**3)

    @property
    def p(self):
        # semi-latus rectum
        return self.a * (1 - self.e**2)

    @property
    def M(self):
        return self.M_0 + self.n * self.epoch


def eci_to_oe(state_eci, mu) -> ArrayLike:
    # Unpack the state vector
    position_eci = state_eci[0:3]
    velocity_eci = state_eci[3:6]

    # Calculate position magnitude and velocity magnitude
    r = jnp.linalg.norm(position_eci)
    v = jnp.linalg.norm(velocity_eci)

    # Calculate specific angular momentum
    h_vector = jnp.cross(position_eci, velocity_eci)
    h = jnp.linalg.norm(h_vector)

    # Calculate inclination (i)
    i = jnp.arccos(h_vector[2] / h)

    # Calculate eccentricity (e)
    e_vector = ((v**2 - mu / r) * position_eci -
                jnp.dot(position_eci, velocity_eci) * velocity_eci) / mu
    e = jnp.linalg.norm(e_vector)

    # Calculate the semimajor axis (a)
    a = 1 / (2 / r - v**2 / mu)

    # Calculate argument of periapsis (omega)
    omega = jnp.arccos(jnp.dot(e_vector, jnp.array([1, 0, 0])) / e)
    if e_vector[1] < 0:
        omega = 2 * jnp.pi - omega

    # Calculate the true anomaly (theta)
    theta = jnp.arccos(jnp.dot(e_vector, position_eci) / (e * r))
    if jnp.dot(position_eci, velocity_eci) < 0:
        theta = 2 * jnp.pi - theta

    # Calculate the right ascension of the ascending node (Omega)
    N_vector = jnp.cross(jnp.array([0, 0, 1]), h_vector)
    N = jnp.linalg.norm(N_vector)
    Omega = jnp.arccos(N_vector[0] / N)
    if N_vector[1] < 0:
        Omega = 2 * jnp.pi - Omega

    return jnp.array([a, e, i, Omega, omega, theta])


def oe_to_eci(oe, theta, mu):
    assert len(oe) == 5  # a, e, i, Omega, omega

    # Unpack the orbital elements
    a, e, i, Omega, omega = oe

    # Calculate the eccentric anomaly (E)
    E = 2 * jnp.arctan(jnp.sqrt((1 - e) / (1 + e)) * jnp.tan(theta / 2))

    # Calculate the mean anomaly (M)
    M = E - e * jnp.sin(E)

    # Calculate the mean motion
    n = jnp.sqrt(mu / a**3)

    # Calculate the eccentric anomaly (E) using Newton's method
    E0 = M
    while True:
        E1 = E0 - (E0 - e * jnp.sin(E0) - M) / (1 - e * jnp.cos(E0))
        if abs(E1 - E0) < 1e-8:
            E = E1
            break
        E0 = E1

    # Calculate the true anomaly (nu)
    nu = 2 * jnp.arctan2(
        jnp.sqrt(1 + e) * jnp.sin(E / 2),
        jnp.sqrt(1 - e) * jnp.cos(E / 2))

    # Calculate the radius (r) and position vector in the orbital plane (in the perifocal frame)
    r = a * (1 - e * jnp.cos(E))
    position_pf = jnp.array([r * jnp.cos(nu), r * jnp.sin(nu), 0])

    # Rotation matrices for orbital elements
    R_omega = jnp.array([[jnp.cos(omega), -jnp.sin(omega), 0],
                         [jnp.sin(omega), jnp.cos(omega), 0], [0, 0, 1]])

    R_i = jnp.array([[1, 0, 0], [0, jnp.cos(i), -jnp.sin(i)],
                     [0, jnp.sin(i), jnp.cos(i)]])

    R_Omega = jnp.array([[jnp.cos(Omega), -jnp.sin(Omega), 0],
                         [jnp.sin(Omega), jnp.cos(Omega), 0], [0, 0, 1]])

    # Transform the position vector to the Earth-centered inertial frame
    position_eci = R_Omega @ (R_i @ (R_omega @ position_pf))

    # Calculate the velocity vector in the orbital plane (in the perifocal frame)
    velocity_pf = jnp.array([-jnp.sin(E),
                             jnp.sqrt(1 - e**2) * jnp.cos(E), 0]) * n * a / r
    # Transform the velocity vector to the Earth-centered inertial frame
    velocity_eci = R_Omega @ (R_i @ (R_omega @ velocity_pf))
    # Return the position and velocity vectors in the Earth-centered inertial frame as a 6-vector
    # return position_eci
    return jnp.array([
        position_eci[0], position_eci[1], position_eci[2], velocity_eci[0],
        velocity_eci[1], velocity_eci[2]
    ])
