import numpy as np
from numpy.typing import NDArray
from . import propagation, conversions
import scipy as sp


# TODO: Implement logging equinoctial orbital elements.
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
    :ivar raan: (Longitude of the right ascending node) Angle between ecliptic 1-axis and nodal vector.
    :ivar argp: (Argument of periapsis) Angle between nodal and eccentricity vectors in the orbital plane.
    :ivar inclination: Angle between orbital and ecliptic planes.
    :ivar true_anomaly: Current location of satellite along the orbit.

    [MODIFIED EQUINOCTIAL ORBITAL ELEMENTS]
    :ivar sl_rectum: 1/2 the length of the orbit's latus-rectum.
    :ivar e_component1:
    :ivar e_component2:
    :ivar n_component1:
    :ivar n_component2:
    :ivar true_latitude: (True longitude) Angle between the ecliptic 1-axis and positon vector.

    [OTHER ORBITAL ELEMENTS]
    :ivar longp: (Longitude of periapsis) Angle between the ecliptic 1-axis and eccentricity vector.
    :ivar argl: (Argument of latitude) Angle between the line of nodes and position vector.

    [OTHER PARAMETERS]
    :ivar spf_angular_momentum: Angular momentum per unit mass of the orbit, a (3, ) vector.
    :ivar eccentricity_vec: The direction of periapsis wrt. the central body, a (3, ) vector.
    :ivar nodal_vec: The direction of the right ascending node wrt. the central body, a (3, ) vector. This lies along
        the line of nodes which marks the intersection between the orbital and ecliptic planes.
    """

    def __init__(
            self,
            position: NDArray[float],
            velocity: NDArray[float],
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
            position: NDArray[float],
            velocity: NDArray[float],
            time,
            grav_param=3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Identical to the __init__() function. Returns an Orbit object based on the position and velocity.
        """

        return cls(position, velocity, time, grav_param)

    @classmethod
    def from_classical_elements(
            cls,
            sm_axis: float,
            eccentricity: float,
            raan: float,
            inclination: float,
            argp: float,
            true_anomaly: float,
            time: float,
            grav_param: float = 3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Alternative constructor which takes in the classical orbital elements and calls elements_2_state() to convert
        to position and velocity and from there generate an Orbit object.
        """

        position, velocity = conversions.classic_elements_2_state(
            sm_axis=sm_axis,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param
        )
        return cls(position, velocity, time, grav_param)

    @classmethod
    def from_classical_elements_p(
        cls,
        sl_rectum: float,
        eccentricity: float,
        raan: float,
        inclination: float,
        argp: float,
        true_anomaly: float,
        time: float,
        grav_param: float = 3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Same as from_classical_elements() but explicitly for parabolic orbits where semi-major axis is not defined and
        semi-latus rectum is used instead.
        """

        position, velocity = conversions.classic_elements_2_state_p(
            sl_rectum=sl_rectum,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param
        )
        return cls(position, velocity, time, grav_param)


    @classmethod
    def from_equinoctial_elements(
            cls,
            sl_rectum: float,
            e_component1: float,
            e_component2: float,
            n_component1: float,
            n_component2: float,
            true_latitude: float,
            time: float,
            grav_param: float = 3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Modified equinoctial version of from_classical_elements().
        """

        position, velocity = conversions.equinoctial_elements_2_state(
            sl_rectum=sl_rectum,
            e_component1=e_component1,
            e_component2=e_component2,
            n_component1=n_component1,
            n_component2=n_component2,
            true_latitude=true_latitude,
        )
        return cls(position, velocity, time, grav_param)

    @classmethod
    def from_gibbs(
            cls,
            position1: NDArray[float],
            position2: NDArray[float],
            position3: NDArray[float],
            time: float,
            current_position_index: int = 2,
            grav_param: float = 3.986004418e14  # Default to Earth in units of m^3/s^2.
    ) -> "Orbit":
        """
        Alternative constructor which returns the position and velocity given three co-planar position vectors. The
        process of determining the conic whose origin lies at the center of three co-planar vectors is known as Gibbs'
        method.

        NOTE: The three positon vectors must be ordered from earliest the latest in time when they are passed as
        arguments in order for Gibbs' method to encode retrograde/prograde correctly.

        [GIBBS METHOD PARAMETERS]
        :param position1:
        :param position2:
        :param position3:
        :param current_position_index: Which position vector the satellite currently is located at.

        [OTHER PARAMETERS]
        :param time:
        :param grav_param:
        """

        # Form the three vectors used in Gibbs' method. The first two correspond to sl_rectum = vec1 / vec2, and the
        # third comes from eccentricity = vec3 / vec2 in the derivation.
        gibbs_vec1 = (
                np.linalg.norm(position3) * np.cross(position1, position2)
                    + np.linalg.norm(position1) * np.cross(position2, position3)
                    + np.linalg.norm(position2) * np.cross(position3, position1)
        )
        gibbs_vec2 = np.cross(position1, position2) + np.cross(position2, position3) + np.cross(position3, position1)
        gibbs_vec3 = (
                (np.linalg.norm(position2) - np.linalg.norm(position3)) * position1
                    + (np.linalg.norm(position3) - np.linalg.norm(position1)) * position2
                    + (np.linalg.norm(position1) - np.linalg.norm(position2)) * position3
        )

        # Compute the velocity corresponding to current position.
        match current_position_index:  # Select current position.
            case 1:
                position = position1
            case 2:
                position = position2
            case 3:
                position = position3
        velocity = (
                1 / np.linalg.norm(position)
                    * np.sqrt(grav_param / (np.linalg.norm(gibbs_vec1) * np.linalg.norm(gibbs_vec2)))
                    * np.cross(gibbs_vec2, position)
                    + np.sqrt(grav_param / (np.linalg.norm(gibbs_vec1) * np.linalg.norm(gibbs_vec2))) * gibbs_vec3
        )

        return cls(position, velocity, time, grav_param)

    @classmethod
    def from_lambert(
            cls,
            position1: NDArray[float],
            position2: NDArray[float],
            tof: float,
            time: float,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            current_position_index: int = 1,
            short_transfer: bool = True,
            prograde: bool = True,
            fg_constraint: bool = True,
            solver_tol=1e-8,
            stumpff_tol=1e-8,
            stumpff_series_length=10,
    ) -> "Orbit":
        """
        A universal variable implementation of Gauss' method to solving Lambert's problem. Lambert's problem involves
        finding the orbit which corresponds to two position vectors and the time-of-flight between them (whether that
        involved traveling the long or short route between them must be specified by the user).
            To do this, the f and g functions and their derivatives are treated as three equations used to solve for
        three unknowns, in this case the Stumpff parameter, universal variable, and a third term termed here as the
        Lambert parameter. These roughly translate to the change in eccentric anomaly, semi-major axis, and semi-latus
        rectum. Since these equations are transcendental in the Stumpff parameter, a root-finding method is used to
        solve them in which a value for the Stumpff parameter is guessed, all three equations are solved and then
        these are plugged into Kepler's equation where time is set equal to the time-of-flight.
            Once the correct value of the parameter is found, the f and g functions are constructed and used to solve
        for the velocity at both points.

        [LAMBERT PROBLEM PARAMETERS]
        :param position1:
        :param position2:
        :param tof: Time-of-flight between the two positions.
        :param current_position_index: Which position vector the satellite currently is located at.
        :param short_transfer: Whether the satellite took the long or short arc between the two positions
        :param prograde: Whether satellite is in prograde, determines the sign on Lambert's constant.
        :param fg_constraint: Whether to compute the gdot-series independently (increasing computation time) or to
            instead use the series constraint (faster but less accurate).
        :param solver_tol: Tolerance to use when solving Kepler's equation.
        :param stumpff_tol: Minimum absolute value of the Stumpff parameter before switching to the infinite series
            definition of the Stumpff series.
        :param stumpff_series_length: How many terms to evaluate in the Stumpff series when using their infinite series
            definitions.

        [OTHER PARAMETERS]
        :param time: Global time at current position vector, not the same as tof.
        :param grav_param:
        """

        # Compute the true anomaly between the two position vectors, then use the short_transfer flag to decide if the
        # short or large arc solution to Lamber's problem should be used.
        true_anomaly = (
                np.arccos(np.dot(position1, position2) / (np.linalg.norm(position1) * np.linalg.norm(position2)))
        )
        if not short_transfer:
            true_anomaly = 2 * np.pi - true_anomaly
        if not prograde:
            true_anomaly *= -1

        # Define a constant to make carrying terms easier.
        lambert_const = (
                np.sqrt(np.linalg.norm(position1) * np.linalg.norm(position2)) * np.sin(true_anomaly)
                / np.sqrt(1 - np.cos(true_anomaly))
        )

        # The UniversalVariablePropagator class contains a method called stumpff_funcs() which given a value of the
        # Stumpff parameter evaluates the Stumpff series. We need this function so we're going to perform a
        # pseudo-instantiation of this class to get access to it, only passing in attributes relevant to calling
        # stumpff_funcs().
        uv_propagator = propagation.universal_variable.UniversalVariablePropagator(
            stumpff_tol=stumpff_tol,
            stumpff_series_length=stumpff_series_length,
        )

        # Solve Lambert's problem using a root-finding method to get the change in the Stumpff parameter between the
        # two positions.
        def eq(x):
            s_func, c_func = uv_propagator.stumpff_funcs(x)
            lambert_param = (
                    np.linalg.norm(position1) + np.linalg.norm(position2)
                        - lambert_const * (1 - x * s_func) / np.sqrt(c_func)
            )
            universal_variable = np.sqrt(lambert_param / c_func)

            return (
                tof - (universal_variable ** 3 * s_func + lambert_const * np.sqrt(lambert_param))
                    / np.sqrt(grav_param)
            )
        stumpff_param = sp.optimize.newton(eq, 0, tol=solver_tol)

        # Compute the f and g functions from the resultant change in the Stumpff parameter.
        s_func, c_func = uv_propagator.stumpff_funcs(stumpff_param)
        lambert_param = (
                np.linalg.norm(position1) + np.linalg.norm(position2)
                - lambert_const * (1 - stumpff_param * s_func) / np.sqrt(c_func)
        )
        universal_variable = np.sqrt(lambert_param / c_func)

        f_func = 1 - universal_variable ** 2 / np.linalg.norm(position1) * c_func
        g_func = tof - universal_variable ** 3 / np.sqrt(grav_param) * s_func
        gdot_func = 1 - universal_variable ** 2 / np.linalg.norm(position2) * c_func
        if fg_constraint:
            fdot_func = (f_func * gdot_func - 1) / g_func
        else:
            fdot_func = (
                    -np.sqrt(grav_param) / (np.linalg.norm(position1) * np.linalg.norm(position2))
                        * universal_variable * (1 - stumpff_param * s_func)
            )

        # Compute the velocity corresponding to current position.
        match current_position_index:
            case 1:
                position = position1
                velocity = (position2 - f_func * position1) / g_func
            case 2:
                position = position2
                if fg_constraint:  # Implicitly uses fg-constraint to eliminate fdot_func and hence velocity1.
                    velocity = (gdot_func * position2 - position1) / g_func
                else:
                    velocity1 = (position2 - f_func * position1) / g_func
                    velocity = fdot_func * position1 + gdot_func * velocity1

        return cls(position, velocity, time, grav_param)

    # ------------------------------
    # ORBITAL ELEMENT UPDATE METHODS
    # ------------------------------
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

    def update_sl_rectum(self):
        self.sl_rectum = np.linalg.norm(self.spf_angular_momentum) ** 2 / self.grav_param

    def update_sm_axis(self):
        self.sm_axis = self.sl_rectum / (1 - self.eccentricity ** 2)

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

    def update_longp(self):
        longp = self.raan + self.argp
        if longp > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            longp -= 2 * np.pi
        self.longp = longp

    def update_argl(self):
        argl = self.argp + self.true_anomaly
        if argl > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            argl -= 2 * np.pi
        self.argl = argl

    def update_true_latitude(self):
        true_latitude = self.raan + self.argp + self.true_anomaly
        if true_latitude > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            true_latitude -= 2 * np.pi
        self.true_latitude = true_latitude


    def update_all(self):
        """
        Master function which updates all the orbital parameters based on the given position and velocity.
        """

        self.update_spf_angular_momentum()
        self.update_eccentricity()
        self.update_nodal_vec()
        self.update_sl_rectum()
        if self.eccentricity == 1:  # Handle parabolic case.
            self.sm_axis = np.inf
        else:
            self.update_sm_axis()
        self.update_raan()
        self.update_inclination()
        self.update_argp()
        self.update_true_anomaly()
        self.update_longp()
        self.update_argl()
        self.update_true_latitude()
