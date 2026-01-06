import numpy as np
from ..dynamics import dcms


def classical_2_state(
        sm_axis: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Converts the classical orbital elements (where true anomaly is the fast parameter) into inertial position and
    velocity.

    Forms the position and velocity components from the semi-major axis, eccentricity, and true anomaly. These are
    then assembled into vectors in a basis fixed to the spacecraft. A 3-3-1-3 rotation sequence via the true anomaly,
    argument of periapsis, inclination, and then RAAN is used to transform these vectors first to the perifocal and
    then to the planet-centered inertial basis. These transformations are performed using DCMs generated via
    :func:`dynamics.euler_2_dcm`.

    Parameters
    ----------
    sm_axis : float
        Semi-major axis.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.

    See Also
    --------
    :func:`classic_2_state_p` : Alternate version of this function which use the semi-latus rectum instead of
        the semi-major axis. Needed for parabolic orbits where the semi-major axis is infinite.
    """

    # Construct the component's of position and velocity in the satellite's local frame.
    sl_rectum = sm_axis * (1 - eccentricity ** 2)
    pos_magnitude = sl_rectum / (1 + eccentricity * np.cos(true_anomaly))  # Trajectory eq.
    pos_magnitude_dt = np.sqrt(grav_param / sl_rectum) * eccentricity * np.sin(true_anomaly)
    true_anomaly_dt = np.sqrt(grav_param * sl_rectum) / pos_magnitude ** 2

    # Construct the DCM from the local to ecliptic frame.
    local_2_perifocal_dcm = dcms.euler_2_dcm(true_anomaly, 3).T
    perifocal_2_inertial_dcm = (
            dcms.euler_2_dcm(raan, 3).T
            @ dcms.euler_2_dcm(inclination, 1).T
            @ dcms.euler_2_dcm(argp, 3).T
    )

    # Compute position and velocity in the local frame and then transform them to the inertial frame.
    position = np.array([pos_magnitude, 0, 0])
    velocity = np.array([pos_magnitude_dt, pos_magnitude * true_anomaly_dt, 0])

    position = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ position
    velocity = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ velocity

    return position, velocity

def state_2_classical(
        position: np.ndarray,
        velocity: np.ndarray,
        grav_param: float = 3.986004418e14
) -> tuple[float, float, float, float, float, float]:
    r"""
    Converts inertial position and velocity into the classical orbital elements (where true anomaly is the fast
    parameter).

    Compute the specific angular momentum, eccentricity, and nodal vectors from the position and velocity. This yields
    the eccentricity which can then be used to compute the semi-major axis via the rearranged trajectory equation.
    Angular geometry using the computed vectors can be used to compute the RAAN, inclination, argument of periapsis,
    and true anomaly via :func:`atan2()`. All but the inclination are then wrapped to :math:`[0, 2\pi]` because
    :func:`atan2()` yields outputs on :math:`[-\pi/2, \pi/2]`.

    Parameters
    ----------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    sm_axis : float
        Semi-major axis.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    See Also
    --------
    :func:`state_2_classical_p` : Alternate version of this function which returns the semi-latus rectum
        instead of the semi-major axis. Useful for parabolic orbits where the semi-major axis is infinite.
    """

    # Define unit vectors.
    unit_vec_1 = np.array([1, 0, 0])
    unit_vec_2 = np.array([0, 1, 0])
    unit_vec_3 = np.array([0, 0, 1])

    # Compute the eccentricity vector and from that the eccentricity.
    spf_angular_momentum = np.cross(position, velocity)
    eccentricity_vec = (
            np.cross(velocity, spf_angular_momentum)
            / grav_param - position / np.linalg.norm(position)
    )
    eccentricity = np.linalg.norm(eccentricity_vec)
    nodal_vec = np.cross(unit_vec_3, spf_angular_momentum)

    # Compute the semi-major axis.
    sl_rectum = np.linalg.norm(spf_angular_momentum) ** 2 / grav_param
    sm_axis = sl_rectum / (1 - eccentricity ** 2)

    # Compute all the angles needed to parameterize an orbit.
    raan = np.arctan2(
        np.dot(nodal_vec, unit_vec_2),
        np.dot(nodal_vec, unit_vec_1)
    )
    inclination = np.arctan2(
        np.dot(spf_angular_momentum, np.cross(nodal_vec, unit_vec_3)),
        np.linalg.norm(nodal_vec) * np.dot(spf_angular_momentum, unit_vec_3)
    )
    argp = np.arctan2(
        np.dot(eccentricity_vec, np.cross(spf_angular_momentum, nodal_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(eccentricity_vec, nodal_vec)
    )
    true_anomaly = np.arctan2(
        np.dot(position, np.cross(spf_angular_momentum, eccentricity_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(position, eccentricity_vec)
    )

    # Wrap angles (not needed for inclination because it is already defined on [-pi/2, pi/2]).
    if raan < 0:  # Wrap to [0, 2pi].
        raan += 2 * np.pi
    if argp < 0:  # Wrap to [0, 2pi].
        argp += 2 * np.pi
    if true_anomaly < 0:  # Wrap to [0, 2pi].
        true_anomaly += 2 * np.pi

    return sm_axis, eccentricity, raan, inclination, argp, true_anomaly

def classical_2_state_p(
        sl_rectum: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
):
    r"""
    Parabolic version of classical_2_state() which takes in the semi-latus rectum instead of the semi-major
    axis.

    For a parabolic orbit the semi-major axis is infinite. As a result computing the semi-latus rectum (and hence the
    orbital radius) from these is impossible and instead to define a parabolic orbit the semi-latus rectum must be
    defined directly. Note that for parabolic orbits only this is equivalent to the radius of periapsis.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.

    See Also
    --------
    :func:`classical_2_state` : Standard version of this function which use the semi-major axis instead of the
        semi-latus rectum.

    Notes
    -----
    The semi-latus rectum :math:`p` is computed from the eccentricity :math:`e` and semi-major axis :math:`a` via

    .. math::

        p = a (1 - e^2)

    For a parabolic orbit this results in the indeterminate form :math:`p = \infty (0)` from which the semi-latus rectum
    can not be recovered.
    """

    # Construct the component's of position and velocity in the satellite's local frame.
    pos_magnitude = sl_rectum / (1 + eccentricity * np.cos(true_anomaly))  # Trajectory eq.
    pos_magnitude_dt = np.sqrt(grav_param / sl_rectum) * eccentricity * np.sin(true_anomaly)
    true_anomaly_dt = np.sqrt(grav_param * sl_rectum) / pos_magnitude ** 2

    # Construct the DCM from the local to ecliptic frame.
    local_2_perifocal_dcm = dcms.euler_2_dcm(true_anomaly, 3).T
    perifocal_2_inertial_dcm = (
            dcms.euler_2_dcm(raan, 3).T
            @ dcms.euler_2_dcm(inclination, 1).T
            @ dcms.euler_2_dcm(argp, 3).T
    )

    # Compute position and velocity in the local frame and then transform them to the inertial frame.
    position = np.array([pos_magnitude, 0, 0])
    velocity = np.array([pos_magnitude_dt, pos_magnitude * true_anomaly_dt, 0])

    position = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ position
    velocity = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ velocity

    return position, velocity

def state_2_classical_p(
        position: np.ndarray,
        velocity: np.ndarray,
        grav_param: float = 3.986004418e14
) -> tuple[float, float, float, float, float, float]:
    r"""
    Parabolic version of state_2_classical() which takes in the semi-latus rectum instead of the semi-major
    axis.

    For a parabolic orbit the semi-major axis is infinite. As a result computing the semi-latus rectum (and hence the
    orbital radius) from these is impossible and instead to define a parabolic orbit the semi-latus rectum must be
    defined directly. Note that for parabolic orbits only this is equivalent to the radius of periapsis.

    Parameters
    ----------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    sl_rectum : float
        Semi-latus rectum.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    See Also
    --------
    :func:`state_2_classical` : Standard version of this function which returns the semi-major axis instead of
        the semi-latus rectum.

    Notes
    -----
    The semi-latus rectum :math:`p` is computed from the eccentricity :math:`e` and semi-major axis :math:`a` via

    .. math::

        p = a (1 - e^2)

    For a parabolic orbit this results in the indeterminate form :math:`p = \infty (0)` from which the semi-latus rectum
    can not be recovered.
    """

    # Define unit vectors.
    unit_vec_1 = np.array([1, 0, 0])
    unit_vec_2 = np.array([0, 1, 0])
    unit_vec_3 = np.array([0, 0, 1])

    # Compute the eccentricity vector and from that the eccentricity.
    spf_angular_momentum = np.cross(position, velocity)
    eccentricity_vec = (
            np.cross(velocity, spf_angular_momentum)
            / grav_param - position / np.linalg.norm(position)
    )
    eccentricity = np.linalg.norm(eccentricity_vec)
    nodal_vec = np.cross(unit_vec_3, spf_angular_momentum)

    # Compute the semi-major axis.
    sl_rectum = np.linalg.norm(spf_angular_momentum) ** 2 / grav_param

    # Compute all the angles needed to parameterize an orbit.
    raan = np.arctan2(
        np.dot(nodal_vec, unit_vec_2),
        np.dot(nodal_vec, unit_vec_1)
    )
    inclination = np.arctan2(
        np.dot(spf_angular_momentum, np.cross(nodal_vec, unit_vec_3)),
        np.linalg.norm(nodal_vec) * np.dot(spf_angular_momentum, unit_vec_3)
    )
    argp = np.arctan2(
        np.dot(eccentricity_vec, np.cross(spf_angular_momentum, nodal_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(eccentricity_vec, nodal_vec)
    )
    true_anomaly = np.arctan2(
        np.dot(position, np.cross(spf_angular_momentum, eccentricity_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(position, eccentricity_vec)
    )

    # Wrap angles (not needed for inclination because it is already defined on [-pi/2, pi/2]).
    if raan < 0:  # Wrap to [0, 2pi].
        raan += 2 * np.pi
    if argp < 0:  # Wrap to [0, 2pi].
        argp += 2 * np.pi
    if true_anomaly < 0:  # Wrap to [0, 2pi].
        true_anomaly += 2 * np.pi

    return sl_rectum, eccentricity, raan, inclination, argp, true_anomaly

def equinoctial_2_state(
        sl_rectum: float,
        e_component1: float,
        e_component2: float,
        n_component1: float,
        n_component2: float,
        true_latitude: float,
        grav_param: float = 3.986004418e14
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Converts the modified equinoctial orbital elements (where true latitude is the fast parameter) into inertial
    position and velocity.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    """

    # Intermediate variables.
    var1 = n_component1 ** 2 - n_component2 ** 2  # alpha
    var2 = 1 + n_component1 ** 2 + n_component2 ** 2  # s
    var3 = 1 + e_component1 * np.cos(true_latitude) + e_component2 * np.sin(true_latitude)  # w
    var4 = sl_rectum / var3  # r

    # Construct position and velocity.
    position = var4 / var2 *  np.array([
        np.cos(true_latitude)
            + var1 * np.cos(true_latitude)
            + 2 * n_component1 * n_component2 * np.sin(true_latitude),
        np.sin(true_latitude)
            - var1 * np.sin(true_latitude)
            + 2 * n_component1 * n_component2 * np.cos(true_latitude),
        2 * (n_component1 * np.sin(true_latitude) - n_component2 * np.cos(true_latitude))
    ])
    velocity = -1 / var2 * np.sqrt(grav_param / sl_rectum) * np.array([
        np.sin(true_latitude) + var1 * np.sin(true_latitude)
            - 2 * n_component1 * n_component2 * np.cos(true_latitude)
            + e_component2 - 2 * e_component1 * n_component1 * n_component2
            + var1 * e_component2,
        -np.cos(true_latitude) + var1 * np.cos(true_latitude)
            + 2 * n_component1 * n_component2 * np.sin(true_latitude)
            - e_component1 + 2 * e_component2 * n_component1 * n_component2
            + var1 * e_component1,
        -2 * (n_component1 * np.cos(true_latitude) + n_component2 * np.sin(true_latitude)
              + e_component1 * n_component1 + e_component2 * n_component2)
    ])

    return position, velocity

def classical_2_equinoctial(
        sm_axis: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Converts the classical orbital elements into the modified equinoctial orbital elements.

    Parameters
    ----------
    sm_axis : float
        Semi-major axis.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.

    Returns
    -------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.

    Notes
    -----
    For a parabolic orbit this function will still run if an infinite semi-major axis is passed in. The resultant
    semi-latus rectum will be NAN but all other parameters will be correct.
    """

    sl_rectum = sm_axis * (1 - eccentricity ** 2)
    e_component1 = eccentricity * np.cos(argp + raan)
    e_component2 = eccentricity * np.sin(argp + raan)
    n_component1 = np.tan(inclination / 2) * np.cos(raan)
    n_component2 = np.tan(inclination / 2) * np.sin(raan)
    true_latitude = raan + argp + true_anomaly

    return sl_rectum, e_component1, e_component2, n_component1, n_component2, true_latitude

def equinoctial_2_classical(
        sl_rectum: float,
        e_component1: float,
        e_component2: float,
        n_component1: float,
        n_component2: float,
        true_latitude: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Converts the modified equinoctial orbital elements to the classical orbital elements.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.

    Returns
    -------
    sm_axis : float
        Semi-major axis.
    eccentricity : float
        Eccentricity.
    raan : float
        Right ascension (longitude) of the ascending node.
    argp: float
        Argument of periapsis.
    inclination : float
        Inclination.
    true_anomaly : float
        True anomaly.
    """

    sm_axis = sl_rectum / (1 - e_component1 ** 2 - e_component2 ** 2)
    eccentricity = np.sqrt(e_component1 ** 2 + e_component2 ** 2)
    inclination = np.arctan2(
        2 * np.sqrt(n_component1 ** 2 + n_component2 ** 2),
        1 - n_component1 ** 2 - n_component2 ** 2
    )
    argp = np.arctan2(
        e_component2 * n_component1 - e_component1 * n_component2,
        e_component1 * n_component1 + e_component2 * n_component2
    )
    raan = np.arctan2(n_component2, n_component1)
    true_anomaly = true_latitude - np.arctan2(e_component2, e_component1)

    return sm_axis, eccentricity, raan, argp, inclination, true_anomaly
