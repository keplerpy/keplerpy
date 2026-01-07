class Time:
    """
    Stores a date and time in UT1 and allows for rapid conversion to other time schemes (including Julian dates and
    Greenwich mean-sidereal time).

    Takes in a date (MM/DD/YYYY) and time (HH::MM::SS.S) and via the use of @property automatically converts to the
    corresponding Julian date and Greenwich mean-sidereal time (GMST). The input time should be in UT1 but technically
    UTC+0 may also be used with approximately 1 s loss in accuracy.

    Attributes
    ----------
    date: str
        Current Gregorian date (MM/DD/YYYY).
    time: str
        Current UT1 time (HH::MM::SS.S).
    """

    def __init__(self, date: str, time: str):
        self.date = date
        self.time = time

    @property
    def julian_date(self):
        """

        """

        return self.julian_date()

    @property
    def gmst(self):
        """

        """

        return self.gmst()
