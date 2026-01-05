import numpy as np


def euler_2_dcm(angle, axis):
    match axis:
        case 1:
            dcm = np.array(
                [[1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)]]
            )
            return dcm
        case 2:
            dcm = np.array(
                [[np.cos(angle), 0, -np.sin(angle)],
                 [0, 1, 0],
                 [np.sin(angle), 0, np.cos(angle)]]
            )
            return dcm
        case 3:
            dcm = np.array(
                [[np.cos(angle), np.sin(angle), 0],
                 [-np.sin(angle), np.cos(angle), 0],
                 [0, 0, 1]]
            )
            return dcm
        case _:
            raise ValueError(f"{axis} is not a valid axis for a Euler angle-based DCM to be generated about.")
