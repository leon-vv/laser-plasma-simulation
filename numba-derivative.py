import numba as n
import time
import numpy as np


x = np.linspace(0, 10, 20)
dx = x[1] - x[0]
y = np.sin(x)

@n.stencil(cval=0.0)
def diff_stencil(y):
    #return 1/dx * (-1/2*y[-1] +0*y[0] + 1/2*y[1])
    return 1/dx * (-1/60*y[-3] + 3/20*y[-2] -3/4*y[-1] + 3/4*y[1] - 3/20*y[2] + 1/60*y[3])


@n.njit(cache=True, inline='always')
def diff(y):
    yd = diff_stencil(y) 

    yd[0] = 1/dx*(-11/6*y[0] + 3 * y[1] - 3/2*y[2] + 1/3*y[3])
    yd[1] = 1/dx*(-1/3*y[0] - 1/2*y[1] + 1*y[2] - 1/6*y[3])
    yd[2] = 1/dx*(1/6*y[0] -1*y[1] + 1/2*y[2]+1/3*y[3])
    
    yd[-1] = 1/dx*(-1/3*y[-4] + 3/2*y[-3] - 3*y[-2] + 11/6*y[-1])
    yd[-2] = 1/dx*(1/6*y[-4] -1*y[-3] + 1/2*y[-2] + 1/3*y[-1])
    yd[-3] = 1/dx*(-1/3*y[-4] -1/2*y[-3] +1*y[-2] -1/6*y[-1])
    
    return yd

@n.stencil(cval=0.0)
def diff2_stencil(y):
    #return 1/dx * (-1/2*y[-1] +0*y[0] + 1/2*y[1])
    return 1/dx**2 * (1/90*y[-3] -3/20*y[-2] + 3/2*y[-1] -49/18*y[0] + 3/2*y[1] -3/20*y[2] + 1/90*y[3])

@n.njit(cache=True, inline='always')
def diff2(y):
    yd = diff2_stencil(y) 
    
    yd[0] = 1/dx**2 *(2*y[0] -5*y[1] + 4*y[2] -1*y[3])
    yd[1] = 1/dx**2 *(2*y[0] -2*y[1] + 1*y[2] + 0*y[3])
    yd[2] = 1/dx**2 *(0*y[0] + 1*y[1] -2*y[2] + 1*y[3])
    
    yd[-1] = 1/dx**2*(-1*y[-4] + 4*y[-3] -5*y[-2] + 2*y[-1])
    yd[-2] = 1/dx**2*(0*y[-4] + 1*y[-3] -2*y[-2] + 1*y[-1])
    yd[-3] = 1/dx**2*(1*y[-4] -2*y[-3] + 1*y[-2] + 0*y[-1])
      
    return yd
    


