import numpy as np
from numpy.typing import NDArray
from dynamics import dcms


class Orbit:
    """
    Class which holds current Cartesian state of the orbit as attributes. Also includes functions to convert between
    orbital elements and a Cartesian state.

    NOTE: Orbital elements are computed once upon instantiation. Recomputing them during propagation can cause them to
    drift so avoid calling them compute_xxx() functions if possible.

    [BASE PARAMETERS]
    :ivar position: A (3, ) vector of the satellite's inertial position.
    :ivar velocity: A (3, ) vector of the satellite's inertial velocity.
    :ivar time: Current time.
    :ivar grav_param: Constant related to the gravitational field strength of the central body.

    [CLASSICAL ORBITAL ELEMENTS]
    :ivar sm_axis: 1/2 length of the orbit's major axis.
    :ivar eccentricity: How elliptical orbit is.
    :ivar raan: Angle between ecliptic 1-axis and nodal vector.
    :ivar argp: Angle between nodal and eccentricity vectors in the orbital plane.
    :ivar inclination: Angle between orbital and ecliptic planes.
    :ivar true_anomaly: Current location of satellite along the orbit.

    [OTHER PARAMETERS]
    :ivar spf_angular_momentum: Angular momentum per unit mass of the orbit, a (3, ) vector.
    :ivar eccentricity_vec: The direction of periapsis wrt. the central body, a (3, ) vector.
    :ivar nodal_vec: The direction of the right ascending node wrt. the central body, a (3, ) vector. This lies along
        the line of nodes which marks the intersection between the orbital and ecliptic planes.
    """

    def __init__(
            self,
            position: NDArray[np.floating],
            velocity: NDArray[np.floating],
            time: float,
            grav_param: float=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ):
        self.position = position
        self.velocity = velocity
        self.time = time
        self.grav_param = grav_param

        # Compute orbital elements along with other useful attributes.
        self.update_all()

    # ---------------------------------
    # ALTERNATE INSTANTIATION FUNCTIONS
    # ---------------------------------
    @classmethod
    def from_state(
            cls,
            position,
            velocity,
            time,
            grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Identical to the __init__() function. Returns an Orbit object based on the position and velocity.
        """

        return cls(position, velocity, time, grav_param)

    @classmethod
    def from_orbital_elements(
            cls,
            sm_axis: float,
            eccentricity: float,
            raan: float,
            inclination: float,
            argp: float,
            true_anomaly: float,
            time: float,
            grav_param: float =3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Alternative constructor which takes in the orbital elements and calls elements_2_state() to convert to position
        and velocity and from there generate an Orbit object.
        """

        position, velocity = cls.elements_2_state(
            sm_axis=sm_axis,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param
        )
        return cls(position, velocity, time, grav_param)

    # -------------------------------
    # ORBITAL ELEMENT UPDATES METHODS
    # -------------------------------
    # NOTE: Unless you know what you are doing just call update_all() because the order these are run in matters.
    def update_spf_angular_momentum(self):
        self.spf_angular_momentum = np.cross(self.position, self.velocity)

    def update_eccentricity(self):  # This one updates eccentricity and eccentricity vector.
        self.eccentricity_vec = (
                np.cross(self.velocity, self.spf_angular_momentum)
                / self.grav_param - self.position / np.linalg.norm(self.position)
        )
        self.eccentricity = np.linalg.norm(self.eccentricity_vec)

    def update_nodal_vec(self):
        unit_vec_3 = np.array([0, 0, 1])
        self.nodal_vec = np.cross(unit_vec_3, self.spf_angular_momentum)

    def update_sm_axis(self):
        parameter = np.linalg.norm(self.spf_angular_momentum) ** 2 / self.grav_param
        self.sm_axis = parameter / (1 - self.eccentricity ** 2)

    def update_raan(self):
        unit_vec_1 = np.array([1, 0, 0])
        unit_vec_2 = np.array([0, 1, 0])
        raan = np.arctan2(
            np.dot(self.nodal_vec, unit_vec_2),
            np.dot(self.nodal_vec, unit_vec_1)
        )
        if raan < 0:  # Wrap to [0, 2pi].
            raan += 2 * np.pi
        self.raan = raan

    def update_inclination(self):
        unit_vec_3 = np.array([0, 0, 1])
        self.inclination = np.arctan2(
            np.dot(self.spf_angular_momentum, np.cross(self.nodal_vec, unit_vec_3)),
            np.linalg.norm(self.nodal_vec) * np.dot(self.spf_angular_momentum, unit_vec_3)
        )

    def update_argp(self):
        argp = np.arctan2(
            np.dot(self.eccentricity_vec, np.cross(self.spf_angular_momentum, self.nodal_vec)),
            np.linalg.norm(self.spf_angular_momentum) * np.dot(self.eccentricity_vec, self.nodal_vec)
        )
        if argp < 0:  # Wrap to [0, 2pi].
            argp += 2 * np.pi
        self.argp = argp

    def update_true_anomaly(self):
        true_anomaly = np.arctan2(
            np.dot(self.position, np.cross(self.spf_angular_momentum, self.eccentricity_vec)),
            np.linalg.norm(self.spf_angular_momentum) * np.dot(self.position, self.eccentricity_vec)
        )
        if true_anomaly < 0:  # Wrap to [0, 2pi].
            true_anomaly += 2 * np.pi
        self.true_anomaly = true_anomaly

    def update_all(self):
        """
        Master function which updates all the orbital parameters based on the given position and velocity.
        """

        self.update_spf_angular_momentum()
        self.update_eccentricity()
        self.update_nodal_vec()
        self.update_sm_axis()
        self.update_raan()
        self.update_inclination()
        self.update_argp()
        self.update_true_anomaly()

    # ---------------------
    # OTHER UTILITY METHODS
    # ---------------------
    @staticmethod
    def elements_2_state(
            sm_axis: float,
            eccentricity: float,
            raan: float,
            argp: float,
            inclination: float,
            true_anomaly: float,
            grav_param: float =3.986004418e14
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
        parameter = sm_axis * (1 - eccentricity ** 2)
        pos_magnitude = parameter / (1 + eccentricity * np.cos(true_anomaly))  # Trajectory eq.
        pos_magnitude_dt = np.sqrt(grav_param / parameter) * eccentricity * np.sin(true_anomaly)
        true_anomaly_dt = np.sqrt(grav_param * parameter) / pos_magnitude ** 2

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
