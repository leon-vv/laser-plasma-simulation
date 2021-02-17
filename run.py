import numpy as np
from numpy.typing import ArrayLike

import analysis as a
import simulate as s
import plotting as p

rho = np.ones_like(s.r)*0.164
ru = np.zeros_like(s.r)

p0 = 170
r_mm = s.r*1e3
pressure = 169*np.exp(-15.5*np.abs(r_mm)**3.24) * 101325 + 101325

start = np.array([rho, ru, pressure])

t_hist, history = s.simulate(start, 0.5e-6, 'spherical')

peak_indices = a.peak_indices(history)

p.combined_video(t_hist, history, peak_indices)






