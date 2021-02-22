import time

import numpy as np
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

N = int(2000)

gamma = 5/3
length = 7e-3 / l_unit # 10 mm

r = np.linspace(0, length, N)
dr: float = r[1] - r[0]
r += dr/2
over_r = 1/r # type: ignore
over_r2 = 1/(r**2) # type: ignore
r2 = r**2

from diff import diff, diff2

assert r.shape == (N,)

@njit(inline='always')
def temperature(u):
    return  4.8093e-4 * u[2] / u[0]

@njit(inline='always')
def viscosity(t):
    return 1.86e-6 * (t**(3/2)) / (t + 2.27e2)  # type: ignore

@njit(inline='always')
def get_damped_viscosity(state, diffu):
    d = dr*diffu
    d_neg = np.where(d < 0, d, 0)
    A = 0.6
    B = 0.1
    C = 0
    D = 0
    return viscosity(temperature(state)) - A*dr*d_neg + 1/2*B*(dr*d_neg)**2 - 1/6*C*(dr*d_neg)**3 - 1/24*D*(dr*d_neg)**4

@njit(cache=True)
def f_cy(state, output):
    (rho, ru, p) = state
    
    u = ru / rho
    
    diffu = diff(u)
     
    mu_damped = get_damped_viscosity(state, diffu)
    
    output[0] = -over_r*diff(r*ru)
    output[1] = -over_r*diff(r * ru**2 / rho) - diff(p) + 2*diff(mu_damped * diffu)
    output[2] = -over_r*diff(r * p*u) - (gamma - 1) * p*over_r*diff(r*u) + 2*(gamma - 1)*mu_damped*diffu**2

def f_cy_wrapper(state):
    output = np.empty_like(state)
    f_cy(state, output)
    return output

@njit(cache=True)
def f_sp(state, output):
    rho, ru, p = state
    u = ru / rho
     
    or2 = over_r2
      
    diffu = diff(u)
    mu_damped = get_damped_viscosity(state, diffu)
      
    output[0] = -or2 * diff(r2 * ru)
    output[1] = -or2 * diff(r2 * ru**2 / rho) - diff(p) + 2*diff(mu_damped * diffu)
    output[2] = -or2 * diff(r2 * p*u) - (gamma - 1) * p*or2*diff(r2*u) + 2*(gamma - 1)*mu_damped*diffu**2


def f_sp_wrapper(state):
    output = np.empty_like(state)
    f_cy(state, output)
    return output


def simulate(start, t_end, symmetry, dt=0.5e-10):
    print("Starting simulation")
    
    t_hist = [0]
    history = [np.copy(start)]
    
    u = np.copy(start)
    t_ = 0
    elapsed = 0
    
    if symmetry == 'spherical':
        f = f_sp_wrapper
    elif symmetry == 'cylindrical':
        f = f_cy_wrapper

    while t_ < t_end:
        start_t = time.perf_counter()
         
        k_1 = dt*f(u) # Fourth order runge kutta
        k_2 = dt*f(u + k_1/2)
        k_3 = dt*f(u + k_2/2)
        k_4 = dt*f(u + k_3)
        u += 1/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        
        # Boundary conditions
        u[0][0] = 18/11*u[0][1] - 9/11*u[0][2] + 2/11*u[0][3]
        u[1][0] = 0
        u[2][0] = 18/11*u[2][1] - 9/11*u[2][2] + 2/11*u[2][3]
         
        if t_ != 0:
            # Do not measure first iteration, as it includes
            # Numba compile time.
            elapsed += time.perf_counter() - start_t
        
        t_ += dt
        
        if len(history) < int(300*t_/t_end):
            print(t_)
            history.append(np.copy(u))
            t_hist.append(t_)
    
    print('Done simulating in %.2f s' % elapsed)
    return np.array(t_hist), np.array(history)
        
