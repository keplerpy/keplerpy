import numpy as np
from dynamics import dcms


# TODO: Add state_2_classical_elements.
def classic_elements_2_state(
        sm_axis: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
):
    """
    Given an orbit described by the classical orbital elements it generates the satellite's current position and
    velocity. Position and velocity are constructed in the satellite's local frame using the elements, and then a
    DCM is used to transform them back into the inertial frame of the central body.
        The @staticmethod decorator is attached so that this may be called before an Orbit object is instantiated
    such as for the from_elements() initializer.

    NOTE: For a description of the input parameters see the docstring for the parent Orbit class.
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

def classic_elements_2_state_p(
        sl_rectum: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
):
    """
    Parabolic version of classical_elements_2_state() which takes in the semi-latus rectum instead of the semi-major
    axis because the latter is undefined.

    NOTE: For a description of the input parameters see the docstring for the parent Orbit class.
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

def equinoctial_elements_2_state(
        sl_rectum: float,
        e_component1: float,
        e_component2: float,
        n_component1: float,
        n_component2: float,
        true_latitude: float,
        grav_param: float = 3.986004418e14
):
    """
    Modified equinoctial version of classic_elements_2_state().

    NOTE: For a description of the input parameters see the docstring for the parent Orbit class.
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
    Converts the classical orbital elements to the modified equinoctial orbital elements.
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
