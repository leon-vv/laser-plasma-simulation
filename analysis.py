import numpy as np

import simulate as s

def volumes_sp():
    return  4/3*np.pi*((s.r + 1/2*s.dr)**3 - (s.r-1/2*s.dr)**3)

def volumes_cy():
    return np.pi*((s.r + 1/2*s.dr)**2 - (s.r - 1/2*s.dr)**2)

def total_energy(state, symmetry):
    rho, ru, p = state
    if symmetry == 'spherical':
        return np.sum(volumes_sp()*(3/2*(p-101325) + 1/2*rho*(ru/rho)**2))
    elif symmetry == 'cylindrical':
        return np.sum(volumes_cy()*(3/2*(p-101325) + 1/2*rho*(ru/rho)**2))

def peak_indices(history):
    return np.array([np.where(s[0] == np.max(s[0]))[0][0] for s in history])

def rankine_hugoniot(history):
    peak = peak_indices(history)
    pressure_at_peak = np.array([history[i][2][peak_indices[i]] for i in range(len(peak_indices))])
    c = np.sqrt(gamma * 101325 / 0.164)
    return c * np.sqrt((gamma + 1) / (2*gamma) * (pressure_at_peak / 101325 - 1) + 1)

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

