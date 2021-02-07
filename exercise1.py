'''
Exercise 1

Fouriertransform of function with a period of 2pi

Dependencies: numpy, matplotlib

Authors:
@Eleonora Svanberg
@Henrik Jörnling
'''

#Modules
import numpy as np
import matplotlib.pyplot as plt

def xarray(N: int) -> np.ndarray:
    '''Array with x values'''
    return np.linspace(0., 2.*np.pi, N, endpoint=False)

def gx(x: np.ndarray) -> np.ndarray:
    '''Gives np array with g(x) values'''
    return 3.*np.sin(2.*x)+2.*np.cos(5.*x)

# a)
N1 = 100 # number of data points
N2 = 50
x1 = xarray(N1)
x2 = xarray(N2)

# b)
gx1 = gx(x1)
gx2 = gx(x2) 

# c)
plt.plot(x1, gx1, label=f'N={N1}') # Plot signal g(x) vs x 
plt.plot(x2, gx2, label=f'N={N2}' )
plt.legend()
plt.show()

# d)
Fgx1 = np.fft.rfft(gx1) # Fouriertransfor F(g) of real signal g(x) (r = real)
Fgx2 = np.fft.rfft(gx2)
print(f'Size of Fgx1 (N={N1}): {len(Fgx1)}')
print(f'Size of Fgx2 (N={N2}): {len(Fgx2)}')

# e)
print(f'c_2={Fgx1[2]:.1f}') #c_2=0.0-150.0j, c_2*N
print(f'c_5={Fgx1[5]:.1f}') #c_5=100.0+0.0j, c_5*N
plt.plot(np.real(Fgx1),'o', color='g', label=f'N={N1}') # Plot real part of Fgx vs looping index 
plt.plot(np.imag(Fgx1),'o', color='r') # Plot imaginary part of Fgx vs looping index
plt.plot(np.real(Fgx2),'x', color='g', label=f'N={N2}')
plt.plot(np.imag(Fgx2),'x', color='r')
plt.title('Complex Fourier')
plt.legend()
plt.show()

plt.plot([2*z/N1 for z in np.real(Fgx1)],'o', color='g', label='a') # Plot real part of Fgx vs looping index 
plt.plot([-2*z/N1 for z in np.imag(Fgx1)],'o', color='r', label='b') # Plot imaginary part of Fgx vs looping index
plt.title(f'Real Forier (N={N1})')
plt.legend()
plt.show()
