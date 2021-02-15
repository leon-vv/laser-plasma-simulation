
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


