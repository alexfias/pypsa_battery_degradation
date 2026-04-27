from __future__ import annotations

import numpy as np


K_T = 6.93e-2
T_REF = 298.0

K_SIGMA = 1.04
SIGMA_REF = 0.5

K_DELTA_1 = 1.40e5
K_DELTA_2 = -0.501
K_DELTA_3 = -1.23e5

K_TIME = 4.14e-10


def dod_stress(delta):
    delta = np.asarray(delta)
    return 1.0 / (K_DELTA_1 * delta**K_DELTA_2 + K_DELTA_3)


def soc_stress(sigma):
    sigma = np.asarray(sigma)
    return np.exp(K_SIGMA * (sigma - SIGMA_REF))


def temperature_stress(temperature_kelvin):
    temperature_kelvin = np.asarray(temperature_kelvin)
    return np.exp(K_T * (temperature_kelvin - T_REF) / (T_REF * temperature_kelvin))


def time_stress(seconds):
    return K_TIME * seconds