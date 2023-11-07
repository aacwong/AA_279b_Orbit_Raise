import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

class Orbit:
    def __init__(self, oe, epoch):
        '''
        an orbit from orbital elements and a time epoch
        '''
        self.a = oe[0]
        self.ecc = oe[1]

class ClohessyWiltshireSys:
    def __init__(self, orbit_ref):
        '''
        input a reference orbit for the CW system
        '''
        n = 10
        self.n = n

    def inertial_to_cw(self, orbit, time_epoch):
        '''
        express an inertial orbit as a state in the CW system
        '''
    
    def cw_to_inertial(self, cw_state, time_epoch):
        '''
        express a CW state as an inertial orbit
        '''

    def state_space(self):
        '''
        returns the state space matrices A,B representing CW dynamics

        x_dot = A*x + B*u
        where:  x = [x, y, z, x_dot, y_dot, z_dot].T
                u = [f_x, f_y, f_z].T

        Returns:
            A: A matrix, 6x6
            B: B matrix, 6x3
        '''
        n = self.n
        A = np.zeros((6,6))
        A[0,3] = 1
        A[1,4] = 1
        A[2,5] = 1
        A[3,0] = 3*n*n
        A[3,4] = 2*n
        A[4,3] = -2*n
        A[5,2] = -n*n
        
        B = np.zeros((6,3))
        B[3,0] = 1
        B[4,1] = 1
        B[5,2] = 1

        return(A,B)


def cw_optimization(x_cw_0, x_cw_f, T_f, cw_ref):
    '''
    convex optimization of approximation of cw equations

    Args:
        x_cw_0: initial state in the cw system
        x_cw_f: final state in the cw system
        T_f: final time to reach x_cw_f
        cw_ref: reference cw system
    
    Returns:
        T: time vector, 6xN+1
        X_T: matrix of state time history, 6xN+2
        U_T: matrix of control inputs, 6xN+1
    '''
    A,B = cw_ref.state_space()
    I = np.eye(6)
    N = 50 # number of control inputs to optimize
    dt = T_f / N
    T = np.linspace(0, T_f, N+1)
    # represents x_1 ... x_N
    X = cvx.Variable(6*(N))
    # u_0 ... u_N
    U = cvx.Variable(3*(N+1))
    # vectorize
    X_k  = cvx.hstack( (x_cw_0, X) )
    X_k1 = cvx.hstack( (X, x_cw_f) )
    A_mat = np.kron(np.eye(N+1,dtype=int), A*dt+I)
    B_mat = np.kron(np.eye(N+1,dtype=int), B*dt)
    
    dyn = X_k1 == A_mat @ X_k + B_mat @ U
    # per-axis thrust constraint
    control_constr = cvx.abs(U) <= .2
    # total thrust vector constraint 
    control_constr = cvx.norm(cvx.reshape(U, [3,N+1]), 2, 0) <= 0.3
    # L-1 norm minimization is minimizing thrust assuming 
    # only axis aligned thrusting
    # L-2 norm minimization assumes rotation independent
    obj = cvx.norm(U,1)
    constr = [dyn, control_constr]

    prob = cvx.Problem(cvx.Minimize(obj), constr)

    prob.solve()
    status = prob.status
    print(status)
    if status == 'infeasible':
        return(-1, [],[],[])

    X_opt = X_k.value
    U_opt = U.value

    X_T = np.hstack((X_opt, x_cw_f))
    return (0, T, X_T.reshape((-1,6)).T, U_opt.reshape((-1,3)).T)

if __name__== '__main__':
    cw_ref = ClohessyWiltshireSys(0)
    cw_ref.n = 0.001
    x_cw_0 = np.array([2,100,2,-2,2,0])
    x_cw_f = np.array([0,0,0,0,0,0])
    T_f = 50
    status, T, X_opt, U_opt = cw_optimization(x_cw_0, x_cw_f, T_f, cw_ref)
    if status == -1:
        print('no trajectory possible')
    else:
        plt.figure()
        plt.plot(X_opt[1,:], X_opt[0,:])
        plt.xlabel('y_cw')
        plt.ylabel('x_cw')
        plt.gca().invert_xaxis()
        plt.figure()
        plt.plot(T, U_opt.T)
        plt.legend(['f_x','f_y','f_z'])
        plt.show()
