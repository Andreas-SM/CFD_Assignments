import numpy as np
import matplotlib.pyplot as plt
#
# set computational parameters
#
L = 1.
N = 40
dx = L/N
U = 4.0
nstep = 1000
dt = 0.0006
Pe_L = 10.
alpha = U*L/Pe_L
#
# print the Courant-Friedrichs-Lewy number (Courant number, or CFL number),
# for a stable time integration we need to satisfy (necessary condition, not sufficient!) |Co| < 1
#
print('Courant Number (< 1?): {}'.format(U*dt/dx))
print('"Diffusive" Courant number (< 0.5?): {}'.format(dt*alpha/dx**2))
print('Cell-based Peclet number (< 2?): {}'.format(U*dx/alpha))
#
# set up grid with ghost cells at both ends; 
# we consider Dirichlet boundary conditions: C[0] = 2*C_L-C[1]; C[N+1] = 2*C_R-C[N-1]
#
x = np.linspace(0. - dx/2, L + dx/2, N + 2)
#
# Numerical solution
#
# initialize `cold` array with initial condition and impose boundary conditions
#
C_L = 1.
C_R = 0.
cold = np.zeros(N+2)
cold[0] = 2*C_L-cold[1]
cold[N+1] = 2*C_R-cold[N]
#
# initialize solution array with zeros
#
c = cold[:]
#
# time iteration loop
#
method = 'FT-CS-CS' # other options: 'FT-CS-CS'
time = 0.
for istep in range(nstep):
    time += dt
    #
    # space finite-differences loop
    #
    if method == 'FT-CS-CS':
        #
        # Euler Forward + Central differences
        #
        for i in range(1, N+1):
            c[i] = cold[i] - dt*U*(cold[i+1]-cold[i-1])/(2*dx) + dt*alpha*(cold[i-1]-2*cold[i]+cold[i+1])/dx**2
        c[0]   = 2*C_L-cold[1]
        c[N+1] = 2*C_R-cold[N]
        cold[:] = c[:]
    elif method == 'FT-UPWIND-CS':
        #
        # Euler Forward + First-order Upwind
        #
        if U >= 0.0:
            for i in range(1, N+1):
                c[i] = cold[i] - dt*U*(cold[i]-cold[i-1])/(dx) + dt*alpha*(cold[i-1]-2*cold[i]+cold[i+1])/dx**2
        else:
            for i in range(1, N+1):
                c[i] = cold[i] - dt*U*(cold[i+1]-cold[i])/(dx) + dt*alpha*(cold[i-1]-2*cold[i]+cold[i+1])/dx**2
        # alternative implementation without ifs:
        #c[i] = cold[i] - dt/dx*(
        #                     (U + abs(U))/2 * (cold[i]  -cold[i-1]) + 
        #                     (U - abs(U))/2 * (cold[i+1]-c[i]     )
        #                    )
        c[0]   = 2*C_L-cold[1]
        c[N+1] = 2*C_R-cold[N]
        cold[:] = c[:]
    else:
        error_message = 'Invalid method: {}'.format(method)
        raise ValueError(error_message)
    #
    # plot solution evolution compared to the initial condition
    #
    i_plot_frequency = 1
    is_plot_reference = True
    if istep % i_plot_frequency == 0:
        plt.plot(x[1:N+1], c[1:N+1])
        if(is_plot_reference):
            plt.plot(x[1:N+1], C_L+(C_R-C_L)*np.exp(Pe_L*x[1:N+1]/L-1)/np.exp(Pe_L-1), '--k', lw=2)
        plt.title("Numerical solution, time =  {:.5f}\n Method: {}; Pe_L = U*L/alpha = {}".format(time, method, Pe_L))
        #plt.pause(0.05)
        plt.clf()
plt.title("Numerical solution, time =  {:.5f}\n Method: {}; Pe_L = U*L/alpha = {}".format(time, method, Pe_L))
plt.plot(x[1:N+1], C_L+(C_R-C_L)*np.exp(Pe_L*x[1:N+1]/L-1)/np.exp(Pe_L-1), '--k', lw=2)
plt.plot(x[1:N+1], c[1:N+1])
plt.show()