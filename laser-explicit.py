import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as s
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib import animation
from numba import njit


# Unit conversion
# Divide to go to simulation units
# Multiply to go back to SI units

l_unit = 1
t_unit = 1
m_unit = 1

rho_unit = m_unit / l_unit**3
ru_unit = rho_unit * l_unit/t_unit
p_unit = (m_unit*l_unit/t_unit**2) / l_unit**3

print('Units: ', rho_unit, ru_unit, p_unit)

N = int(4000)

gamma = 5/3
length = 7e-3 / l_unit # 10 mm

r = np.linspace(0, length, N)
dr = r[1] - r[0]
r += dr/2
over_r = 1/r
over_r2 = 1/(r**2)
r2 = r**2

assert r.shape == (N,)

dt = 0.5e-10/t_unit # 0.1 us
t_end = 1e-6/t_unit # 5 us

rho = np.ones_like(r)*0.164 / rho_unit
ru = np.zeros_like(r) / ru_unit
p0 = 170
r_mm = r*1e3
#p = (p0*101325*np.exp(-(r/ 1e-3)**2) + 101325) / p_unit
p = (169*np.exp(-15.5*np.abs(r_mm)**3.24) * 101325 + 101325) / p_unit

start = np.array([rho, ru, p])
u = np.copy(start)
history = [np.copy(u)]
t_hist = [0]

def diff(f):
    #assert D1.shape == (N, N)
    #assert f.shape == (N,)
    return np.gradient(f, dr, edge_order=1)
    #return D1.dot(f)

def diff2(f):
    #assert D2.shape == (N, N)
    #assert f.shape == (N,)
    return diff(diff(f))
    #return D2.dot(f)

def volumes():
    return  4/3*np.pi*((r + 1/2*dr)**3 - (r-1/2*dr)**3)

def total_energy(state):
    rho, ru, p = state
    return np.sum(volumes()*(3/2*(p-101325) + 1/2*rho*(ru/rho)**2))

@njit
def temperature(u):
    return  4.8093e-4 * u[2] / u[0]

@njit
def viscosity(t):
    return 1.86e-6 * (t**(3/2)) / (t + 2.27e2) 

@njit(cache=True)
def get_damped_viscosity(state, diffu):
    d = dr*diffu
    d_neg = np.where(d < 0, d, 0)
    A = 2
    B = 1
    C = 0
    D = 0
    return viscosity(temperature(state)) - A*dr*d_neg + 1/2*B*(dr*d_neg)**2 - 1/6*C*(dr*d_neg)**3 - 1/24*D*(dr*d_neg)**4

def f_cy(state):
    (rho, ru, p) = state
    
    u = ru / rho
     

    diffu = diff(u)
    
    mu_damped = get_damped_viscosity(state, diffu)
    
    drho = -over_r*diff(r*ru)
    dru = -over_r*diff(r * ru**2 / rho) - diff(p) + 2*diff(mu_damped * diffu)
    dp = -over_r*diff(r * p*u) - (gamma - 1) * p*over_r*diff(r*u) + 2*(gamma - 1)*mu_damped*diffu**2

    return np.array([drho, dru, dp])

@njit(cache=True)
def f_sp_numba(state, diffs, mu):
    rho, ru, p = state
    u = ru / rho
    
    gamma = 5/3
     
    or2 = over_r2
     
    diffu = diffs[0]
     
    drho = -or2 * diffs[1]
    dru = -or2 * diffs[2] - diffs[3] + 2*diffs[4]
    dp = -or2 * diffs[5]  - (gamma - 1) * p*or2*diffs[6] + 2*(gamma - 1)*mu*diffu**2
    
    return (drho, dru, dp)

def f_sp(state):
    rho, ru, p = state
    u = ru / rho

    diffu = diff(u)
     
    mu = get_damped_viscosity(state, diffu)
    
    diffs = np.array([
        diffu,
        diff(r2 * ru),
        diff(r2 * ru**2 / rho),
        diff(p),
        diff(mu * diffu),
        diff(r2*p*u),
        diff(r2*u)])
    
    return np.array(f_sp_numba(state, diffs, mu))

def simulate(history):
    u = np.copy(start)
    t_ = 0
    elapsed = 0
     
    while t_ < t_end:
        start_t = time.perf_counter()
        
        f = f_sp
        
        k_1 = dt*f(u) # Fourth order runge kutta
        k_2 = dt*f(u + k_1/2)
        k_3 = dt*f(u + k_2/2)
        k_4 = dt*f(u + k_3)
        u += 1/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        #u += dt*f(u)

        # Boundary conditions
        u[0][0] = u[0][1] # drho/dr = 0
        u[1][0] = 0
        u[2][0] = u[2][0] # dp/dr = 0
         
        #u[0] = u[1]
        #u[-1] = u[-2]
          
        if t_ != 0:
            # Do not measure first iteration, as it includes
            # Numba compile time.
            elapsed += time.perf_counter() - start_t
        
        t_ += dt
        
        if len(history) < int(1500*t_/t_end):
            print(t_)
            history.append(np.copy(u))
            t_hist.append(t_)

    print('Done simulating in %.2f s' % elapsed)
        
simulate(history)

print('Visualizing')


def combined_video(st, peak_indices, save_as=None):
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams["animation.html"] = "jshtml"
    
    pressure = [s[2] for s in st]
    density = [s[0] for s in st]
    temps = [temperature(s) for s in st]
    vel = [s[1] / s[0] for s in st]
     
    
    fig, axs = plt.subplots(4, sharex=True, figsize=(12, 10))
    #fig.tight_layout()
    
    t = fig.text(0.17, 0.92, '0.00 us', fontsize=25)
        
    l1, = axs[0].plot(r_mm, pressure[0]/1e5, label="Pressure", color="red")
    axs[0].set_ylabel("Pressure (bar)", color="red")
    axs[0].set_ylim(0, 10)
    
    vl = axs[0].axvline(r_mm[peak_indices[0]], ls='-', color='y')
     
    l2, = axs[1].plot(r_mm, density[0], label="Density", color="blue")
    axs[1].set_ylim(np.min(density), np.max(density) * 1.1)
    axs[1].set_ylabel("Density (kg/m^3)", color="blue")
    axs[1].set_xlabel("Distance (mm)")
    axs[1].legend(fontsize=20)
     
    l3, = axs[2].plot(r_mm, temps[0], label="Temperature", color="black")
    axs[2].set_ylim(0, np.max(temps) * 1.1)
    axs[2].set_ylabel("Temperature (K)")

    l4, = axs[3].plot(r_mm, vel[0])
    axs[3].set_ylabel('Velocity (m/s)')
    axs[3].set_ylim(np.min(vel) * 1.1, np.max(vel) * 1.1)
      
    def animate(i):
        l4.set_data(r_mm, vel[i])
        l3.set_data(r_mm, temps[i])
        l2.set_data(r_mm, density[i])
        l1.set_data(r_mm, pressure[i]/1e5)
        vl.set_xdata([r_mm[peak_indices[i]], r_mm[peak_indices[i]]])
        t.set_text("%.2f us" % (t_hist[i] * 1e6))

    #total_frames = len(st)
    #frames = range(total_frames) if frames == None else np.linspace(0, total_frames-1, frames).astype(np.int)
    interval = (30000*t_hist[-1]*1e6) / len(st)
        
    video_combined = animation.FuncAnimation(fig, animate, interval=5000/len(st), frames=len(st))
    
    if save_as != None:
        video_combined.save(save_as)
    
    plt.rcParams.update({'figure.autolayout': False})
    plt.show()
    
    return video_combined


def plot_all(hist):
    fig, ax = plt.subplots()
    plt.xlabel('Distance (m)')
    plt.ylabel('Density (kg/m^3)')
    t = fig.text(0.7, 0.75, '0.00 us', fontsize=15)
    line1, = ax.plot(r, hist[0][0], label="With shock damping")
    plt.ylim(0, 1.5*np.max(hist[-1][0]))

    plt.legend()
     
    def init():
        line1.set_data([], [])
        return line1,
     
    def animate(i):
        line1.set_data(r, hist[i][0])
        t.set_text('%.2f us' % (t_hist[i]*1e6))
        
        return line1,
     
    anim = animation.FuncAnimation(fig, animate, interval=5000/len(hist), init_func=init, frames=len(hist))
    #anim.save('density.mp4')
    
    plt.show()

peak_indices = [np.where(s[0] == np.max(s[0]))[0][0] for s in history]
combined_video(history, peak_indices)#save_as='spherical-%i.mp4' % N)
plt.show()

plt.figure()
energies = [total_energy(u) for u in history]
plt.plot(np.array(t_hist)*1e6, energies)
plt.xlabel("Time (us)")
plt.ylabel("Energy (J)")
plt.ylim(0, 2*np.max(energies))

plt.figure()
peak = r[peak_indices]

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

peak_filtered = moving_average(peak, 40)


plt.title('Position of shock front')
plt.ylabel('Distance (mm)')
plt.plot(np.array(t_hist)*1e6, peak_filtered*1000)
plt.plot(np.array(t_hist)*1e6, peak*1000)
plt.xlabel('Time (us)')

pressure_at_peak = np.array([history[i][2][peak_indices[i]] for i in range(len(peak_indices))])

plt.figure()
plt.title('Pressure at peak')
plt.plot(t_hist, pressure_at_peak)
plt.ylim(0, 5*101325)
plt.show()

plt.figure()
plt.title("Speed of schok front")
c = np.sqrt(gamma * 101325 / 0.164)
print('Speed of sound: ', c)


rankine = c * np.sqrt((gamma + 1) / (2*gamma) * (pressure_at_peak / 101325 - 1) + 1)

plt.plot(np.array(t_hist)*1e6, np.gradient(np.array(peak_filtered), t_hist))
plt.plot(np.array(t_hist)*1e6, rankine)

plt.ylim(0, 7e3)
plt.xlabel("Time (us)")
plt.ylabel("Speed (m/s)")


plt.show()




