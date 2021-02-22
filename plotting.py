import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import analysis as a
import simulate as s

def combined_video(t_hist, st, peak_indices, symmetry, save_as=None):
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams["animation.html"] = "jshtml"
    
    assert len(t_hist) == len(st)
    assert len(st) == len(peak_indices)
    
    r_mm = s.r * 1000
    
    pressure = [s[2] for s in st]
    density = [s[0] for s in st]
    temps = [s.temperature(u) for u in st]
    vel = [s[1] / s[0] for s in st]
    
    fig, axs = plt.subplots(4, sharex=True, figsize=(12, 10))
    #fig.tight_layout()
    
    t = fig.text(0.77, 0.95, 'Time: 0.00 us', fontsize=15)
    e = fig.text(0.77, 0.92, 'Energy change: 0.0 %', fontsize=15)
        
    l1, = axs[0].plot(r_mm, pressure[0]/1e5, label="Pressure", color="red")
    axs[0].set_ylabel("Pressure (bar)", color="red")
    axs[0].set_ylim(0, np.max(pressure[0]) / 1e5)
    
    vl = axs[0].axvline(r_mm[peak_indices[0]], ls='-', color='y')
     
    l2, = axs[1].plot(r_mm, density[0], label="Density", color="blue")
    axs[1].set_ylim(0, np.max(density) * 1.1)
    axs[1].set_ylabel("Density (kg/m^3)", color="blue")
     
    l3, = axs[2].plot(r_mm, temps[0], label="Temperature", color="black")
    axs[2].set_ylim(0, np.max(temps) * 1.1)
    axs[2].set_ylabel("Temperature (K)")

    l4, = axs[3].plot(r_mm, vel[0])
    axs[3].set_ylabel('Velocity (m/s)')
    axs[3].set_ylim(np.min(vel) * 1.1, np.max(vel) * 1.1)
      
    axs[3].set_xlabel("Distance (mm)")
    
    def animate(i):
        l4.set_data(r_mm, vel[i])
        l3.set_data(r_mm, temps[i])
        l2.set_data(r_mm, density[i])
        l1.set_data(r_mm, pressure[i]/1e5)
        vl.set_xdata([r_mm[peak_indices[i]], r_mm[peak_indices[i]]])
        t.set_text("Time: %.2f us" % (t_hist[i] * 1e6))
        e_change = (1 - a.total_energy(st[i], symmetry)/a.total_energy(st[0], symmetry)) * 100
        e.set_text('Energy change: %.1f %%' % e_change)
    
    #total_frames = len(st)
    #frames = range(total_frames) if frames == None else np.linspace(0, total_frames-1, frames).astype(np.int)
    dt = (t_hist[-1] - t_hist[0]) / len(t_hist) / 1e-6 * 30000 # 30 Second per us
    print(dt, len(t_hist))
    video_combined = animation.FuncAnimation(fig, animate, interval=dt, frames=len(st), cache_frame_data=False)
    
    if save_as != None:
        video_combined.save(save_as)
    
    plt.rcParams.update({'figure.autolayout': False})
    plt.show()
    
    return video_combined

def energy_plot(t_hist, energies):
    plt.figure()
    plt.plot(np.array(t_hist)*1e6, energies)
    plt.xlabel("Time (us)")
    plt.ylabel("Energy (J)")
    plt.ylim(0, 2*np.max(energies))
    plt.show()







