'''
    File: solve_NSE_liddrivencavity_with_pm.py
    Update: 31.08.2020
    Author: Allen Drews (allen.m.drews@durham.ac.uk,   allen.drews@web.de)

    Description:
    2D vorticity stream-function solver for the
    Navier-Stokes equations for an incompressible fluid.
    The solution is computed using the projection method (pm)
    with finite difference discretisation on a staggered grid as
    detailed in B. Seibold, A compact and fast Matlab code solving the
    incompressible Navier-Stokes equations on rectangular domains,
    2008, MIT .
 .
    
    Implemented problem: Lid-driven cavity

    Requirements:
    - Libraries: matplotlib, numpy, scipy.fft
    - For creating a 'results' directory: makedirs, path, EEXIST
    - For tracking the computation time: time 
    
'''
from numpy import linalg as LA
import numpy as np
from matplotlib import cm, rc, rcParams, pyplot as plt
from scipy.fft import dst, idst, dct, idct
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
RN = 500            # Reynolds number
U0 = 1              # characteristic velocity 
Ls = 1              # characteristic length scale 
nu = U0*Ls/RN       # viscosity
Utop = U0           # velocity at lid
rho = 1             # density 

# Domain (horizontal/ vertical max/min, grid block separation)
xmax = 1
ymax = 1
xmin = 0
ymin = 0
dx = (xmax-xmin) / nx
dy = (ymax-ymin) / ny

### Staggered grid ###
# Gridpoints (cell centres)
xcentre = (np.arange(1,nx+1)-0.5)*dx
ycentre = (np.arange(1,ny+1)-0.5)*dy

# Gridpoints (cell corners)
xcorner = np.arange(0,nx+1)*dx
ycorner = np.arange(0,ny+1)*dy

# Gridpoints (centre of cell edges)
xuedge = (np.arange(-dx,1+dx,dx))+dx/2
yuedge = (np.arange(0,1+dy,dy))

xvedge = (np.arange(0,1+dx,dx))
yvedge = (np.arange(0-dy,1+dy,dy))+dy/2

# Grid
X, Y = np.meshgrid(ycentre, xcentre)
Xco, Yco = np.meshgrid(ycorner, xcorner)
Xuedge, Yuedge = np.meshgrid(xuedge, yuedge)
Xvedge, Yvedge = np.meshgrid(xvedge, yvedge)

### Time parameters ###
# T: time needed to converge close to the steady-state solution
if 1/nu <= 100:             # low Re-flow
    T = 10.0
elif 100 < 1/nu <= 800:     # medium Re-flow
    T = 22.0
elif 800 < 1/nu <= 1500:    # medium-high Re-flow
    T = 30.0
else:                       # high Re-flow
    T = 70.0
    
# dt: time-step, optimised for grid-size 128x128
if 1/nu < 400:
    dt = 1.0e-2
elif 400 <= 1/nu < 1200:
    dt = 8.0e-3
else:
    dt = 5.0e-3         
    
nt = int(T/dt) + 1
print("Evolution time: ", nt*dt)
print("Time-step: ", dt)

# Initiate velocity (u,v) and non-linear terms (Nu/Nv = div((u,v)*(u,v)))
u = np.zeros((nx+1, ny+2))  # horizontal velocity 
v = np.zeros((nx+2, ny+1))  # vertical velocity

Nu_old = np.zeros((nx-1, ny))  # horizontal non-linear component
Nv_old = np.zeros((nx, ny-1))  # vertical non-linear component

### Implicit treatment ###
# Set boundary conditions for viscous terms (Lu/Lv = Laplace((u,v)))
Luy = (1/dy**2) * ( u[1:-1, 0:-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2:] )
Lubc = np.zeros(np.shape(Luy))
Lubc[0 ,:] = 0 / dx**2      # Left
Lubc[-1,:] = 0 / dx**2      # Right
Lubc[:,-1] = 2*Utop / dy**2 # Top
Lubc[:, 0] = 0 / dy**2      # Bot

Lvy = (1/dy**2) * ( v[1:-1, 0:-2] - 2*v[1:-1, 1:-1] + v[1:-1, 2:] )
Lvbc = np.zeros(np.shape(Lvy))
Lvbc[0, :] = 0 / dx**2      # Left
Lvbc[-1,:] = 0 / dx**2      # Right
Lvbc[:,-1] = 0 / dy**2      # Top
Lvbc[:, 0] = 0 / dy**2      # Bot

### Poisson/ Helmholtz equation ###
# For solving the Poisson equation, pre-compute Fourier coefficients
h = dx
kx = np.arange(nx)
ky = np.arange(ny)
Fx = (1/dx**2) * 2 * (np.cos(np.pi * kx / nx) - 1) # Fourier transform in x (DCT)
Fy = (1/dy**2) * 2 * (np.cos(np.pi * ky / ny) - 1) # Fourier transform in y (DCT)
MWX, MWY = np.meshgrid(Fy, Fx)  # Fourier coefficients

# For solving the Helmholtz equation, pre-compute Fourier coefficients
kxu = (np.arange(1, nx)).T
kyu = (np.arange(1, ny+1))
Fxu = (1/dx**2) * 2 * (np.cos(np.pi * kxu / nx) - 1) # Fourier transform in x (DST)
Fyu = (1/dy**2) * 2 * (np.cos(np.pi * kyu / ny) - 1) # Fourier transform in y (DST)
Fxu = Fxu.reshape(nx-1,1)
Fu = Fxu + Fyu               # Fourier coefficients

kxv = (np.arange(1, nx+1))
kyv = (np.arange(1, ny)).T
Fxv = (1/dx**2) * 2 * (np.cos(np.pi * kxv / nx) - 1) # Fourier transform in x (DST)
Fyv = (1/dy**2) * 2 * (np.cos(np.pi * kyv / ny) - 1) # Fourier transform in y (DST)
Fyv = Fyv.reshape(ny-1,1)
Fv = Fxv + Fyv               # Fourier coefficients

print("Calculating u and v for Re = %g on a %gx%g grid..." %(RN,nx,ny))

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
    
    ### Boundary conditions ###
    # Left
    u[0,:] = 0
    v[0,:] = 0 - v[1,:]   
    
    # Right
    u[-1,:] = 0
    v[-1,:] = 0 - v[-2,:]
    
    # Bottom
    u[:, 0] = 0 - u[:, 1]  
    v[:, 0] = 0

    # Top
    u[:,-1] = 2*Utop - u[:,-2]  
    v[:,-1] = 0
    
    
    ### Viscous terms Lu = nabla^2(u) ###
    Lux = (1/dx**2)*( u[0:-2, 1:-1] - 2*u[1:-1, 1:-1] + u[2:, 1:-1] )
    Luy = (1/dy**2)*( u[1:-1, 0:-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2:] )
    
    Lvx = (1/dx**2)*( v[0:-2, 1:-1] - 2*v[1:-1, 1:-1] + v[2:, 1:-1] )
    Lvy = (1/dy**2)*( v[1:-1, 0:-2] - 2*v[1:-1, 1:-1] + v[1:-1, 2:] )

    Lu = Lux + Luy
    Lv = Lvx + Lvy
    
    ### Convective terms Nu = nabla(u*u) ###
    #1. Interpolate velocities 
    ucentre = 0.5*( u[0:-1, 1:-1] + u[1:, 1:-1] )
    ucorner = 0.5*( u[:, 0:-1] + u[:, 1:] )
    vcorner = 0.5*( v[0:-1, :] + v[1:, :] )
    vcentre = 0.5*( v[1:-1, 0:-1] + v[1:-1, 1:] )

    #2. Multiply to get quadratic and mixed terms
    uuce = np.multiply(ucentre,ucentre)     # quadratic term u^2
    uvco = np.multiply(ucorner,vcorner)     # mixed term uv
    vvce = np.multiply(vcentre,vcentre)     # quadratic term v^2

    #3. Define the non-linear convective terms
    Nu = ((1/dx)*( uuce[1:, :] - uuce[0:-1, :] )
          + (1/dy)*( uvco[1:-1, 1:] - uvco[1:-1, 0:-1] ))
    Nv = ((1/dy)*( vvce[:, 1:] - vvce[:, 0:-1] )
          + (1/dx)*( uvco[1:, 1:-1] - uvco[0:-1, 1:-1] ))
    
    ### Solving for intermediate velocity using fast Fourier method ###
    # Right-hand side of Poisson equation
    rhsu = u[1:-1, 1:-1] - dt*((3*Nu - Nu_old)/2 - (Ls*U0/(2*RN)) * (Lu + Lubc))
    rhsv = v[1:-1, 1:-1] - dt*((3*Nv - Nv_old)/2 - (Ls*U0/(2*RN)) * (Lv + Lvbc))
    
    # Compute the discrete Fourier transform of the RHS
    intmu = dst(dst(rhsu, type=1, axis=0), type=2, axis=1)
    intmv = dst(dst(rhsv, type=2, axis=0), type=1, axis=1)
    
    # Solve in Fourier space
    intmu = non_zero_division(intmu, (1 - (dt*Ls*U0/(2*RN)) * Fu))
    intmv = non_zero_division(intmv, (1 - (dt*Ls*U0/(2*RN)) * Fv).T)

    # Invert the Fourier transform 
    intmu = idst(idst(intmu, type=1, axis=0), type=2, axis=1) 
    intmv = idst(idst(intmv, type=2, axis=0), type=1, axis=1)

    # Update the velocity on the interior
    u[1:-1, 1: -1] = intmu
    v[1:-1, 1: -1] = intmv
    
    # Update the non-linear term
    Nu_old = Nu
    Nv_old = Nv

    ### Correct the intermediate velocity with pressure ###                  
    # Right-hand side of Poisson equation
    b = ((1/dx) * (u[1:, 1:-1] - u[0:-1,1:-1])
         + (1/dy) * (v[1:-1,1:] - v[1:-1,0:-1])) 

    # Compute the discrete Fourier transform of the RHS
    p = dct(dct(b, type=2, axis=1, norm='ortho'),
            type=2, axis=0, norm='ortho')
    
    # Solve in Fourier space
    p = non_zero_division(p, MWX + MWY)
    p[0][0] = 0                 # to ensure a unique solution
    
    # Invert the Fourier transform (normalisation handled by "norm=ortho")
    p = idct(idct(p, type=2, axis=0, norm='ortho'),
             type=2, axis=1, norm='ortho')
    
    ### Compute the new divergence-free velocity field ###
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - (1/rho) * (1/dx) * ( p[1:, :] - p[0:-1, :] )
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (1/rho) * (1/dy) * ( p[:, 1:] - p[:, 0:-1] )

    # Dynamic output: plot the velocity field
    if t%100==0:
     
        ### Get velocity at cell centre (for visualisation)
        ucentre = ( u[0:-1, 1:-1] + u[1:, 1:-1] ) / 2
        vcentre = ( v[1:-1, 0:-1] + v[1:-1, 1:] ) / 2
    
        cp = plt.contourf(Y, X, np.sqrt(ucentre**2+vcentre**2), resolution, cmap=cm.coolwarm)
        cbar = fig.colorbar(cp, ticks=[0.0,0.12,0.25,0.37,0.5, 0.62,0.75,0.87,1.0])
        cbar.set_label('Velocity', rotation=270,  labelpad=20)
        plt.quiver(Y[::space, ::space], X[::space, ::space],
                   ucentre[::space, ::space], vcentre[::space, ::space],
                   color='white',linewidths=0.1)
        plt.title(r'Re = %g, t = %gs' %(RN, t*dt))
        plt.xlabel('x')
        plt.ylabel('y')
        fig.canvas.draw()
        plt.pause(dt)
        plt.clf()
        
t1 = time.time()
print("Time for computation: %g s" %(t1-t0))

### Check that (u,v) is divergence free
divu = ((1/dx)*( u[1:, 1:-1] - u[0:-1, 1:-1] )
     + (1/dy)*( v[1:-1, 1:] - v[1:-1, 0:-1] ))
print("div(u): ", LA.norm(divu))   
        
### Plot and save results
output_directory = "results-pm-solver"
mkdir_p(output_directory)
print("Plotting and saving results...")

# Get velocity at cell centre (for visualisation)
ucentre = ( u[0:-1, 1:-1] + u[1:, 1:-1] ) / 2
vcentre = ( v[1:-1, 0:-1] + v[1:-1, 1:] ) / 2

plt.title(r'Re = %g, t = %gs' %(RN, t*dt))
plt.xlabel('x')
plt.ylabel('y')
cp = plt.contourf(Y, X, np.sqrt(ucentre**2 + vcentre**2), resolution, cmap=cm.coolwarm)
cbar = plt.colorbar(cp, ticks=[0.0,0.12,0.25,0.37,0.5, 0.62,0.75,0.87,1.0])
cbar.set_label('Velocity', rotation=270, labelpad=20)
plt.savefig(str(output_directory) + '/Re%g_%gx%g_velocity.png' %(RN,nx,ny), bbox_inches='tight')
plt.show()
plt.clf()

# Get vorticity at cell corners (for visualisation)
vx = (v[1:, :] - v[0:-1, :])/dx
uy = (u[:, 1:] - u[:, 0:-1])/dy
omega = vx - uy

plt.title(r'Vorticity, Re = %g, t = %gs' %(RN, t*dt))
plt.xlabel('x')
plt.ylabel('y')
cp = plt.contour(Yco, Xco, omega, 3*resolution, colors='k')
plt.clabel(cp, inline=1, fontsize=8)
plt.savefig(str(output_directory) + '/Re%g_%gx%g_vorticity.png' %(RN,nx,ny), bbox_inches='tight')
plt.show()

print("Done!")
