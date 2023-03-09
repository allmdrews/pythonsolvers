'''
    File: solve_NSE_kelvinhelmholtzinstability_with_vsf.py
    Update: 31.08.2020
    Author: Allen Drews (allen.m.drews@durham.ac.uk,   allen.drews@web.de)

    Description:
    2D vorticity stream-function solver for the
    Navier-Stokes equations for an incompressible fluid.
    The solution is computed using the vorticity stream-function (vsf)
    method with an FTCS finite difference discretisation as detailed in
    A. Salih, Streamfunction-vorticity Formulation, 2013, IIST .
    
    Implemented problem: Kelvin-Helmholtz instability
    Initial condition as given in Schroeder et al. 2018, see
    https://arxiv.org/pdf/1803.06893.pdf .

    Optionally, an initial condition like in part III/IV of the lecture
    notes (p.26) can be specified instead. 

    Requirements:
    - Packages: matplotlib, numpy, scipy.fft
    - For creating a 'results' directory: makedirs, path, EEXIST
    - For tracking the computation time: time 
    
'''

from numpy import linalg as LA
import numpy as np
from matplotlib import cm, rc, rcParams, pyplot as plt
from scipy.fft import dst, idst, fft, ifft
from errno import EEXIST
from os import makedirs,path
import time

def non_zero_division(n, d):
    '''For division in Fourier space; ensures there is no division by zero.'''
    return np.divide(n, d, out=np.zeros_like(n), where=d!=0)

def mkdir_p(mypath):
    '''Creates a directory; equivalent to using mkdir -p on the command line'''
    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
        
print("Defining variables and setting up grid...")
### Declare variables ###        
nx = 128            # grid points in x
ny = 128            # grid points in y

# Flow parameters
RN = 300            # Reynolds number
U0 = 1              # characteristic velocity 
Ls = 1./28.         # characteristic length scale 
nu = U0*Ls/RN       # viscosity
Utop = U0           # velocity in upper half of domain
Ubot = -U0          # velocity in lower half of domain

# Domain (horizontal/ vertical max/min, grid block separation)
xmax = 1
ymax = 1
xmin = 0
ymin = 0
dx = (xmax-xmin) / nx
dy = (ymax-ymin) / ny

# Grid
xvalues = np.linspace(xmin, xmax, nx+1) 
yvalues = np.linspace(ymin, ymax, ny+1)
X, Y = np.meshgrid(yvalues, xvalues)

### Time parameters ###
# step dt optimised for grid-size 128x128
T = 5.0
if 1/nu <= 100:             # low Re-flow
    dt = (RN/2) * dx**2*dy**2/(dx**2+dy**2)        #100% of stability criterion
elif 100 < 1/nu <= 800:     # medium Re-flow
    dt = 0.8 * (RN/2) * dx**2*dy**2/(dx**2+dy**2)  #65% of stability criterion
elif 800 < 1/nu <= 1500:    # medium-high Re-flow
    dt = 0.2 * (RN/2) * dx**2*dy**2/(dx**2+dy**2)  #20% of stability criterion
elif 1500 < 1/nu <= 4000:   # high Re-flow
    dt = 0.05 * (RN/2) * dx**2*dy**2/(dx**2+dy**2) #5% of stability criterion
else:
    dt = 1.0e-4             # adjust as needed for higher Re
    
nt = int(T/dt) + 1
print("Evolution time: ", nt*dt, "s")
print("Time-step: ", dt, "s")

# Initiate vorticity (omega), stream function (psi) and velocity (u,v) 
omega = np.zeros((nx+1, ny+1))  # vorticity
psi = np.zeros((nx+1, ny+1))    # stream function

# Shear layer option 1 #################################################
# tanh-shaped initial condition as given in Schroeder et al. 2018

delta0 = Ls     # characteristic length scale
cn = 1E-3       # scale factor

u = (np.tanh((2*X-1)/delta0)
     + cn*((2/(delta0**2))*(0.5-X)*np.exp(-((X-0.5)/delta0)**2)*
           (np.cos(8*np.pi*Y) + np.cos(20*np.pi*Y))
           )
     )

v = cn*np.exp(-((X-0.5)/delta0)**2)*(np.sin(8*np.pi*Y)*8*np.pi
                                          + np.sin(20*np.pi*Y)*20*np.pi)
          
# Initial vorticity       
omega = ((16*cn*np.pi**2)*np.exp(-((X-0.5)/delta0)**2)*(4*np.cos(8*np.pi*Y)
                                               + 25*np.cos(20*np.pi*Y))
     +(2*cn/(delta0**2))*np.exp(-((X-0.5)/delta0)**2)*(np.cos(8*np.pi*Y)
                                               + np.cos(20*np.pi*Y))
     -(4*cn/(delta0**4))*(X-0.5)**2*np.exp(-((X-0.5)/delta0)**2)*(np.cos(8*np.pi*Y)
                                               + np.cos(20*np.pi*Y))
     -(2/delta0)*(1/np.cosh((2*X-1)/delta0))**2
     )

# Shear layer option 2 #################################################
# Uncomment to enable. Note that this case has not been tested against literature values
'''
u = np.zeros((nx+1, ny+1))  # horizontal velocity
v = np.zeros((nx+1, ny+1))  # vertical velocity

Utop = U0
Ubot = 0

u[:,int(nx/2):] = Utop 
u[:,0:-int(nx/2)] = Ubot

alpha = 10.
beta = 2.

for i in range(nx):
    for j in range(ny):
        if j > int(ny/2) - alpha*np.cos(beta*np.pi*i/nx): 
            u[i][j] = Utop
        elif j < int(ny/2) - alpha*np.cos(beta*np.pi*i/nx): 
            u[i][j] = Ubot      

Ls = 1/beta      
nu = U0*Ls/RN
cn = 1E-3       

omega[1:-1, 1:-1] = ((v[2:, 1:-1] - v[0:-2,1:-1])/(2*dx)
                 - (u[1:-1, 2:] - u[1:-1,0:-2])/(2*dy))

omega[:, 0] = omega[:, 1] 
omega[:,-1] = omega[:,-2]
omega[0, :] = omega[-2,:]
omega[-1,:] = omega[1, :]

omegan = np.empty_like(omega)
psin = np.empty_like(psi)
'''
# For solving the Poisson equation, pre-compute Fourier coefficients
# (can be precomputed to save computation time)
kx = np.arange(0,nx)
ky = np.arange(1,ny)
Fx = (1/dx**2) * 2 * (np.cos(2 * np.pi * kx / nx) - 1)  # Fourier transform in x
Fy = (1/dy**2) * 2 * (np.cos(np.pi * ky / ny) - 1)      # Fourier transform in y
M, N = np.meshgrid(Fy, Fx)  # Fourier coefficients

print("Calculating \u03C9 and \u03C8 for Re = %g on a %gx%g grid..." %(RN,nx,ny))

# Configuring the plot environment
rcParams['font.family'] = "serif"
rcParams['lines.linewidth'] = 0.5
rcParams['contour.negative_linestyle'] = 'solid'

if RN <= 800: 
    resolution = int(nx/2)      # number of contourlines drawn in plot
elif 1500 < RN <= 4000: 
    resolution = int(3*nx/2)
else:
    resolution = nx
    
space = 8               # display every 8th grid point

fig = plt.figure(figsize=(9,7), dpi=100)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x')
plt.ylabel('y')

t0 = time.time()        # for tracking the computation time
# Main time loop
for t in range(nt+1):
    omegan = omega.copy()
    psin = psi.copy()
    
    ### Compute omega using FTCS scheme ###
    uij = u[1:-1, 1:-1]
    vij = v[1:-1, 1:-1]
    omega[1:-1, 1:-1] = ( omegan[1:-1, 1:-1] +
                      dt*( - uij * ((omegan[2:, 1:-1] - omegan[0:-2,1:-1]) / (2*dx))
                           - vij * ((omegan[1:-1, 2:] - omegan[1:-1,0:-2]) / (2*dy))
                           + nu * ((omegan[2:, 1:-1] - 2*omegan[1:-1, 1:-1] + omegan[0:-2,1:-1]) / (dx**2)
                                   + (omegan[1:-1, 2:] - 2*omegan[1:-1, 1:-1] + omegan[1:-1,0:-2]) / (dy**2)))
                      )

    ### Boundary conditions for omega ###
    # Left, right
    # Explicitly periodic along the horizontal (FTCS scheme)   
    omega[0, 1:-1] = ( omegan[0, 1:-1] +
                      dt*( - u[0, 1:-1] * ((omegan[1,1:-1] - omegan[-2,1:-1]) / (2*dx))
                           - v[0, 1:-1] * ((omegan[0, 2: ] - omegan[0, 0:-2]) / (2*dy))
                           + nu * ((omegan[1, 1:-1] - 2*omegan[0, 1:-1] + omegan[-2,1:-1]) / (dx**2)
                                   + (omegan[0, 2:] - 2*omegan[0, 1:-1] + omegan[0, 0:-2]) / (dy**2)))
                      )
    
    
    omega[-1,1:-1] = omega[0, 1:-1]

    # Bottom, top 
    omega[:, 0] = omega[:, 1] 
    omega[:,-1] = omega[:,-2]

    ### Solve Poisson equation for stream function using fast Fourier method ###
    # Along vertical: use only interior points (Dirichlet B.C.s)
    # Along horizontal: all but rightmost point (periodic B.C.s)
    # Right-hand side of Poisson equation
    q = -omega[0:-1,1:-1]
    
    # Compute the discrete Fourier transform of the RHS
    psi = fft(dst(q, type=1, axis=1, norm=None), axis=0)
    
    # Solve in Fourier space
    psi = non_zero_division(psi, M+N)
    
    # Invert the Fourier transform
    psi = np.real(idst(ifft(psi, axis=0), type=1, axis=1, norm=None))
    
    # Free-slip vertical boundary
    psi = np.concatenate((psi[:,0:1],psi,psi[:,-1:]), axis=1)
    
    # Periodic horizontal boundary
    psi = np.concatenate((psi,psi[0:1,:]), axis=0)
    
    ### Update the velocities ###
    # Interior points
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2 * dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2 * dx)
    
    # Boundary conditions
    # Left, right
    u[-1,1:-1] = (psi[0, 2:] - psi[0, 0:-2]) / (2 * dy)
    v[-1,1:-1] = -(psi[1,1:-1] - psi[-2,1:-1]) / (2 * dx)
    
    u[0, 1:-1] = u[-1,1:-1]
    v[0, 1:-1] = v[-1,1:-1]

    # Bottom, top
    u[:, 0] = Ubot 
    u[:,-1] = Utop 
    v[:, 0] = v[:, 1] 
    v[:,-1] = v[:,-2]
    
    # Dynamic output: plot the velocity field on top of the vorticity
    if t%300==0:
        vortp = plt.contourf(Y, X, -np.sqrt(omega**2), resolution, cmap=cm.jet)
        plt.colorbar(vortp).set_label('Vorticity', rotation=270, labelpad=20)
        plt.quiver(Y[::space, ::space], X[::space, ::space],
                   u[::space, ::space], v[::space, ::space],
                   color='white',linewidths=0.1)
        plt.title(r'Re = %g, t = %gs' %(RN, t*dt))
        plt.xlabel('x')
        plt.ylabel('y')
        fig.canvas.draw()
        plt.pause(dt*cn)
        plt.clf()
           
t1 = time.time()
print("Done!")
print("Time for computation: %g s" %(t1-t0))

### Check that (u,v) is divergence free
divu = ((u[0:-2,1:-1] - u[2:,1:-1]) / (2*dx)
             + (v[1:-1,0:-2] - v[1:-1,2:]) / (2*dy))
print("div(u): ", LA.norm(divu))

### Plot and save results
output_directory = "results-vsf-solver"
mkdir_p(output_directory)
print("Plotting and saving results...")

plt.title(r'Vorticity, Re = %g, t = %gs' %(RN, t*dt))
plt.xlabel('x')
plt.ylabel('y')
vortp = plt.contourf(Y, X, -np.sqrt(omega**2), resolution, cmap=cm.jet)
plt.colorbar(vortp).set_label('Vorticity', rotation=270, labelpad=20)
plt.savefig(str(output_directory) + '/Re%g_%gx%g_vorticity_khi.png' %(RN,nx,ny), bbox_inches='tight')
plt.show()
plt.clf()

plt.title(r'Re = %g, t = %gs' %(RN, t*dt))
plt.xlabel('x')
plt.ylabel('y')
cp = plt.contourf(Y, X, np.sqrt(u**2 + v**2), resolution, cmap=cm.coolwarm)
cbar = plt.colorbar(cp)
cbar.set_label('Velocity', rotation=270, labelpad=20)
plt.savefig(str(output_directory) + '/Re%g_%gx%g_velocity_khi.png' %(RN,nx,ny), bbox_inches='tight')
plt.show()
plt.clf()

plt.title(r'Stream-function, Re = %g, t = %gs' %(RN, t*dt))
plt.xlabel('x')
plt.ylabel('y')
cp = plt.contour(Y, X, psi, 2*space, colors='k')
plt.clabel(cp, inline=1, fontsize=8)
plt.savefig(str(output_directory) + '/Re%g_%gx%g_streamfunction_khi.png' %(RN,nx,ny), bbox_inches='tight')
plt.show()      

print("Done!")
