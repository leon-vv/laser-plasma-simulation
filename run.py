import analysis as a
import simulate as s
import plotting as p

rho = np.ones_like(r)*0.164 / rho_unit
ru = np.zeros_like(r) / ru_unit
p0 = 170
r_mm = r*1e3
#p = (p0*101325*np.exp(-(r/ 1e-3)**2) + 101325) / p_unit
p = (169*np.exp(-15.5*np.abs(r_mm)**3.24) * 101325 + 101325) / p_unit

start = np.array([rho, ru, p])

t_hist, history = simulate(start, 1e-6, 'cylindrical')

peak_indices = a.peak_indices(history)

p.combined_video(history, peak_indices)






