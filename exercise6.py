'''
Exercise 6

Analysis of unit pulses (dirac functions)

Dependencies: numpy, matplotlib

Authors:
@Eleonora Svanberg
@Henrik Jörnling
'''

#Modules
import numpy as np
import matplotlib.pyplot as plt

def plotfft(pt: np.ndarray, n: int) -> None:
    '''Plot proporties of F(pt)'''
    Fft = np.fft.rfft(pt) # Fouriertransform F(f) of pulse
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.suptitle(f'n={n}')
    ax1.plot(Fft, color='r')# Plot real part of Fft vs looping index 
    ax1.set_title('$Fft$')
    ax2.plot(np.real(Fft), color='b') # Plot real part of Fft vs looping index 
    ax2.set_title('$Re(F)$')
    ax3.plot(np.imag(Fft), color='g') # Plot real part of Fft vs looping index 
    ax3.set_title('$Im(F)$')
    ax4.plot(np.abs(Fft), color='orange') # Plot real part of Fft vs looping index 
    ax4.set_title('$|F|$')
    plt.show()
    

def sample(T1:int) -> np.ndarray:
    '''
    Returns a data sample by given time
    '''
    N = 1024 # number of data points
    T0 = 0.0 # start of sample [s]
    t = np.linspace(T0, T1, N, endpoint=False) # array with sample time
    Ts = T1-T0
    deltat = Ts/N
    vs = 1/deltat
    deltav = 1/Ts
    print(f't[0]={t[0]:.2f}, t[1]={t[1]:.2f}, t[2]={t[2]:.2f}')
    print(f'Ts={Ts}, N={N}, deltat={deltat:.2f}, vs={vs:.2f}, deltav={deltav:.2f}')

    return t

T1 = 1.0
t = sample(T1)
pt1, pt2 = np.zeros(t.shape), np.zeros(t.shape)

# a)
pt1[0] = 1
plotfft(pt1, 0)

# b)
pt2[20] = 1
plotfft(pt2, 20)
