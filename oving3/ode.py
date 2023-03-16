import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
plt.ion()

def euler(f, t0, y0, tend, N=100):
    '''
    Euler's method for solving y'=f(t,y), y(t0)=y0.
    '''
    h = (tend-t0)/N         # Stepsize

     
    # In the case of a scalar ODE, convert y0 to a numpy vector.
    if not isinstance(y0, np.ndarray): 
        y0 = np.array([y0])
        m = 1
    else:
        m = len(y0)
    
    # Make arrays for storing the solution. 
    ysol = np.zeros((N+1, m))
    tsol = np.zeros(N+1)
    # Insert the initial values
    ysol[0,:] = y0
    tsol[0] = t0

    tn = t0
    yn = y0

    # Main loop
    for n in range(N):
        # One step of Euler's method
        yn = yn+h*f(tn,yn)
        tn = tn+h

        # Store the solution
        ysol[n+1,:] = yn
        tsol[n+1] = tn

    # In case of a scalar problem, convert the solution into a 1d-array
    if m==1:
        ysol = ysol[:,0] 

    return tsol, ysol
# end of euler


def heun_euler(f, t0, y0, tend, h0, tol=1.e-6):
    '''
    Heun-Eulers's adaptive method for solving y'=f(t,y), y(t0)=y0.
    '''
     
    # In the case of a scalar problem, convert y0 to a numpy vector.
    if not isinstance(y0, np.ndarray): 
        y0 = np.array([y0])
        m = 1
    else:
        m = len(y0)
   
    ysol = np.array([y0]) # Arrays to store the solution
    tsol = np.array([t0])

    yn = y0
    tn = t0
    h = h0
    MaxNumberOfSteps = 10000  # Maximum number of steps, accepted and rejeced
    ncount = 0

    # Main loop
    while tn < tend - 1.e-10:
        if tn+h > tend:
            h = tend - tn
        
        # One step with Heun's method
        k1 = f(tn,yn)
        k2 = f(tn+h, yn+h*k1)

        # Calculate the error estimate
        le_n = 0.5*h*np.linalg.norm(k2-k1)   # Measured in the 2-norm
        
        
        if le_n <= tol:             
            # Solution accepted, update tn and yn
            yn = yn+0.5*h*(k1+k2)
            tn = tn+h
            # Store the solution
            ysol = np.concatenate((ysol, np.array([yn])))
            tsol = np.append(tsol, tn)
        # else the step is rejected and nothing is updated. 

        # Change the stepsize
        h = 0.8*(tol/le_n)**(1/2)*h
        
        ncount += 1
        if ncount > MaxNumberOfSteps:
            raise Exception('Maximum number of steps reached')
  
    # In case of a scalar problem, convert the solution into a 1d-array
    if m==1:
        ysol = ysol[:,0] 

    return tsol, ysol
# end of heun_euler




def ode_example1():
    # Numerical example 1
    # Define the problem to be solved
    def f(t,y):    
        # The right hand side of y' = f(t,y)
        return -2*t*y

    # Set the initial value
    y0 =   1            
    t0, tend = 0, 1   

    # Number of steps
    N = 10

    # Solve the equation numerically
    tsol, ysol = euler(f, t0, y0, tend, N=N)

    # Plot the numerical solution together with the exact, if available
    texact = np.linspace(0,1,1001)
    plt.plot(tsol, ysol, 'o', label='Euler')
    plt.plot(texact, np.exp(-texact**2), '--', label='Exact')
    plt.legend()
    plt.xlabel('t');
    plt.figure(2) # only for the python file

    # Plot the error 
    error = np.abs(ysol-np.exp(-tsol**2))
    plt.semilogy(tsol, error, 'o--')
    plt.title('Error')
    print(f'Max error is {np.max(error):.2e}')
# end of ode_example1


def ode_example2():
    # Numerical example 2
    # Define the problem to be solved
    def lotka_volterra(t,y):    
        # The right hand side of y' = f(t,y)
        alpha, beta, delta, gamma = 2, 1, 0.5, 1
        dy = np.zeros(2)
        dy[0] = alpha*y[0]-beta*y[0]*y[1]
        dy[1] = delta*y[0]*y[1] - gamma*y[1]
        return dy

    # Set the initial value
    y0 = np.array([2.0, 0.5])           
    t0, tend = 0, 20   

    # Number of steps
    N = 1000

    # Solve the equation numerically
    tsol, ysol = euler(lotka_volterra, t0, y0, tend, N=N)

    # Plot the numerical solution i
    plt.plot(tsol, ysol)
    plt.legend(['y1','y2'])
    plt.xlabel('t');
# end of ode_example2

def ode_example3():
    # Numerical example 3
    # Define the problem to be solved
    def f(t,y):    
        # The right hand side of y' = f(t,y)
        return -2*t*y

    # Set the initial value
    y0 =   1            
    t0, tend = 0, 1   

    # Number of steps
    N = 10

    # Solve the equation numerically 
    # with Euler's method
    t_euler, y_euler = euler(f, t0, y0, tend, N=N)
    # And with Heun's method
    t_heun, y_heun = heun(f, t0, y0, tend, N=N//2) 


    # Plot the numerical solution together with the exact, if available
    plt.plot(t_euler, y_euler, 'o', label='Euler')
    plt.plot(t_heun, y_heun, 'd', label='Heun')
    texact = np.linspace(0,1,1001)
    plt.plot(texact, np.exp(-texact**2), '--', label='Exact')
    plt.legend()
    plt.xlabel('t');
    plt.figure(2) # only for the python file ex3

    # Plot the errors for both methods
    error_euler = np.abs(y_euler-np.exp(-t_euler**2))
    error_heun = np.abs(y_heun-np.exp(-t_heun**2))
    plt.semilogy(t_euler, error_euler, 'o--', label='Euler')
    plt.semilogy(t_heun, error_heun, 'd--', label='Heun')
    plt.title('Error')
    print(f'Max error for Euler is {np.max(error_euler):.2e} and for Heun {np.max(error_heun):.2e}')
    # end of error plot ex3
    print('\n\n')

    # Print the error as a function of h
    y_exact = np.exp(-tend**2)
    N = 10
    print('Error in Euler and Heun\n')
    print('h           Euler       Heun')
    for n in range(10):
        t_euler, y_euler = euler(f, t0, y0, tend, N=N)
        t_heun, y_heun = heun(f, t0, y0, tend, N=N//2) 
        error_euler = np.abs(y_exact-y_euler[-1])
        error_heun = np.abs(y_exact-y_heun[-1])
        print(f'{(tend-t0)/N:.3e}   {error_euler:.3e}   {error_heun:.3e}')
        N = 2*N
# end of ode_example3

def ode_example4():
    plt.figure(1)
    # Numerical example 4
    f = lambda t,y: -2*t*y
    t0, tend = 0, 1
    y0 = 1
    tol = 1.e-3
    h0 = 0.1
    tsol, ysol = heun_euler(f, t0, y0, tend, h0, tol=tol)
    plt.plot(tsol, ysol, 'o--')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Numerical solution y(t)');

    # split 41  (This for automatic inclusion in the jupyter note-book)
    plt.figure(2) 

    # Plot the error from the adaptive method
    error = np.abs(np.exp(-tsol**2) - ysol)
    plt.semilogy(tsol, error, '.-')
    plt.title('Error in Heun-Euler for dy/dt=-2ty')
    plt.xlabel('t');
    plt.ylabel('|e_n|')

    # split 42
    plt.figure(3)

    # Plot the step size sequence
    h_n = np.diff(tsol)            # array with the stepsizes h_n = x_{n+1} 
    t_n = tsol[0:-1]            # array with x_num[n], n=0..N-1
    plt.semilogy(t_n, h_n, '.-')
    plt.xlabel('t')
    plt.ylabel('h')
    plt.title('Stepsize variations');

# end of ode_example4


#===  Stiff ODEs ===
def implicit_euler(A, g, t0, y0, tend, N=100):
    '''
    Implicit Eulers's method for solving y'=Ay+g(t), y(t0)=y0,
    '''
    h = (tend-t0)/N         # Stepsize

     
    # In the case of a scalar problem, convert y0 to a numpy vector.
    if not isinstance(y0, np.ndarray): 
        y0 = np.array([y0])
        m = 1
    else:
        m = len(y0)
    
    # Make arrays for storing the solution. 
    ysol = np.zeros((N+1, m))
    ysol[0,:] = y0
    tsol = np.zeros(N+1)
    tsol[0] = t0

    # Main loop
    M = np.eye(m)-h*A
    for n in range(N):
        yn = ysol[n,:]
        tn = tsol[n]
        
        # One step with implicit Euler
        b = yn + h*g(tn)                # b = y + hf(t_{n+1})
        ynext = solve(M, b)         # Solve M y_next = b
        tnext = tn+h

        ysol[n+1,:] = ynext
        tsol[n+1] = tnext
  
    # In case of a scalar problem, convert the solution into a 1d-array
    if m==1:
        ysol = ysol[:,0] 

    return tsol, ysol
# end of implicit_euler

def trapezoidal_ieuler(A, g, t0, y0, tend, h0, tol=1.e-6):
    '''
    Trapezoidal - implicit euler adaptive method for solving y'=f(t,y), y(t0)=y0.
    '''
     
    # In the case of a scalar problem, convert y0 to a numpy vector.
    if not isinstance(y0, np.ndarray): 
        y0 = np.array([y0])
        m = 1
    else:
        m = len(y0)
   
    ysol = np.array([y0]) # Arrays to store the solution
    tsol = np.array([t0])

    yn = y0
    tn = t0
    h = h0
    MaxNumberOfSteps = 10000  # Maximum number of steps, accepted and rejeced
    ncount = 0

    # Main loop
    while tn < tend - 1.e-10:
        if tn+h > tend:
            h = tend - tn

        # One step with implicit Euler
        M = np.eye(m)-h*A
        b = yn + h*g(tn+h)
        y_ie = solve(M, b)
        
        # One step with the trapezoidal rule
        M = np.eye(m)-0.5*h*A
        b = yn + 0.5*h*np.dot(A,yn) + 0.5*h*(g(tn)+g(tn+h))
        y_trap = solve(M, b)  

        le_n = norm(y_trap-y_ie)
        
        if le_n <= tol:             
            # Solution accepted, update tn and yn
            yn = y_trap
            tn = tn+h
            # Store the solution
            ysol = np.concatenate((ysol, np.array([yn])))
            tsol = np.append(tsol, tn)
        # else the step is rejected and nothing is updated. 

        # Change the stepsize
        h = 0.8*(tol/le_n)**(1/2)*h
        
        ncount += 1
        if ncount > MaxNumberOfSteps:
            raise Exception('Maximum number of steps reached')
  
    # In case of a scalar problem, convert the solution into a 1d-array
    if m==1:
        ysol = ysol[:,0] 

    return tsol, ysol
# end of trapezoidal_ieuler


def ode_example_1s():
    # Numerical example 1s
    def f(t, y):
        a = 2
        dy = np.array([-2*y[0]+y[1]+2*np.sin(t),
                    (a-1)*y[0]-a*y[1]+a*(np.cos(t)-np.sin(t))])
        return dy

    # Initial values and integration interval 
    y0 = np.array([2, 3])
    t0, tend = 0, 10
    h0 = 0.1

    tol = 1.e-2
    # Solve the ODE using different tolerances 
    for n in range(3):
        print('\nTol = {:.1e}'.format(tol)) 
        tsol, ysol = heun_euler(f, t0, y0, tend, h0, tol)
        
        if n==0:
            # Plot the solution
            plt.subplot(2,1,1)
            plt.plot(tsol, ysol)
            plt.ylabel('y')
            plt.subplot(2,1,2)

        # Plot the step size control
        plt.semilogy(tsol[0:-1], np.diff(tsol), label='Tol={:.1e}'.format(tol));
        
        tol = 1.e-2*tol         # Reduce the tolerance by a factor 0.01.
    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend();
# end of ode_example_1s

def ode_example_2s():
    # Numerical example 2s
    def f(t, y):
        # y' = f(x,y) = A*y+g(x)
        a = 9
        dy = np.array([-2*y[0]+y[1]+2*np.sin(t),
                    (a-1)*y[0]-a*y[1]+a*(np.cos(t)-np.sin(t))])
        return dy

    # Startverdier og integrasjonsintervall 
    y0 = np.array([2, 3])
    t0, tend = 0, 10

    N = 50
    tsol, ysol = euler(f, t0, y0, tend, N=N)
    plt.plot(tsol, ysol);
# end of ode_example_2s

def ode_example_3s():
    # Numerical example 3s
    a = 9
    A = np.array([[-2, 1],[a-1, -a]])
    g = lambda t: np.array([2*np.sin(t), a*(np.cos(t)-np.sin(t))])

    # Initial values and integration interval 
    y0 = np.array([2, 3])
    t0, tend = 0, 10
    N = 50

    tsol, ysol = implicit_euler(A, g, t0, y0, tend, N)
    plt.plot(tsol, ysol);
# end of ode_example_3s

def ode_example_4s():
    # Numerical example 4s
    a = 9
    A = np.array([[-2, 1],[a-1, -a]])
    g = lambda t: np.array([2*np.sin(t), a*(np.cos(t)-np.sin(t))])

    # Initial values and integration interval 
    y0 = np.array([2, 3])
    t0, tend = 0, 10
    h0 = 0.1

    tol = 1.e-2
    # Solve the ODE using different tolerances 
    for n in range(3):
        print('\nTol = {:.1e}'.format(tol)) 
        tsol, ysol = trapezoidal_ieuler(A, g, t0, y0, tend, h0, tol=tol)

        
        if n==0:
            # Plot the solution
            plt.subplot(2,1,1)
            plt.plot(tsol, ysol)
            plt.ylabel('y')
            plt.subplot(2,1,2)

        # Plot the step size control
        plt.semilogy(tsol[0:-1], np.diff(tsol), label='Tol={:.1e}'.format(tol));
        
        tol = 1.e-2*tol         # Reduce the tolerance by a factor 0.01.
    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend();
# end of ode_example_4s
