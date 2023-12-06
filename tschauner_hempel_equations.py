import numpy as np
import math
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.linalg import block_diag
import copy

class Orbit:
    def __init__(self, oe, mu, epoch):
        '''
        an orbit from orbital elements and a time epoch
        '''
        self.a = oe[0] # sma
        self.ecc = oe[1] # ecc
        self.incl = oe[2] # inclination
        self.raan = oe[3] # raan
        self.arg_peri = oe[4] # arg peri
        self.nu = oe[5] # true anomaly
        self.mu = mu
        self.epoch = epoch # time at which the true anomaly is given (sec)

    def toPosVel(self):
        '''
        calculate the position and velocity (ECI) corresponding to the given oe 
        '''
        norm = np.linalg.norm
        cnu = math.cos(self.nu)
        snu = math.sin(self.nu)
        a = self.a
        n = self.n
        p = self.p
        ecc = self.ecc

        one_e_cnu = 1+ecc*cnu

        E = np.arccos( (ecc+cnu)/one_e_cnu )
        if self.nu>math.pi:
            E = -E + 2*math.pi
        cE = math.cos(E)
        sE = math.sin(E)

        R_ref = p/one_e_cnu
        # perifocal
        pos_pqw = R_ref*np.array([cnu,snu,0]).T
        vel_pqw = a*n/(1-ecc*cE) * np.array([-sE, math.sqrt(1-ecc*ecc)*cE, 0]).T
        dcm_peri_to_eci = self.R_peri_to_eci
        pos = dcm_peri_to_eci @ pos_pqw
        vel = dcm_peri_to_eci @ vel_pqw
        return pos,vel

    @property
    def period(self):
        return 2*math.pi/self.n
    @property
    def n(self):
        return math.sqrt(self.mu/self.a**3)
    @property
    def p(self):
        return self.a*(1-self.ecc*self.ecc)
    @property
    def R_peri_to_eci(self):
        def R_1(a):
            ca = np.cos(a)
            sa = np.sin(a)
            return np.array([[1,0,0],
                             [0,ca,sa],
                             [0,-sa,ca]])
        def R_3(a):
            ca = np.cos(a)
            sa = np.sin(a)
            return np.array([[ca,sa,0],
                             [-sa,ca,0],
                             [0,0,1]])
        return R_3(-self.raan)@R_1(-self.incl)@R_3(-self.arg_peri)

    def matchNu(self, pos):
        '''
        calculate the true anomaly at a given position
        relative to this orbit
        '''
        # given position expressed in perifocal
        dcm_eci_to_peri = self.R_peri_to_eci.T
        pos_pqw = dcm_eci_to_peri @ pos
        # true anomaly of pos
        nu_pos = math.acos(pos_pqw[0]/np.linalg.norm(pos_pqw))
        if pos_pqw[1] < 0:
            nu_pos = 2*math.pi - nu_pos
        return nu_pos
    
    def trueAnomToTime(self, nu):
        '''
        converts from true anomaly to a time value

        a true anomaly matching the true anomaly of the orbit eph will be equal to 
        the time epoch of the orbit
        '''
        nus = np.hstack([self.nu, nu])
        ecc = self.ecc
        cnu = np.cos(nus)
        E = np.arccos( (ecc+cnu)/(1+ecc*cnu) )
        # correct for pi wrapping
        s1 = np.floor(nus/math.pi)
        s2 = np.floor(nus/math.pi/2+.5)
        ecc_anom = (-1)**s1 * E + s2*2*math.pi
        mean_anom = ecc_anom-ecc*np.sin(ecc_anom)
        T = (mean_anom-mean_anom[0])/self.n + self.epoch
        return T[1:]

class TschaunerHempelSys:
    def __init__(self, orbit_ref):
        '''
        input a reference orbit for the TH system
        '''
        self.orbit_ref = orbit_ref

    def state_space(self,nu):
        '''
        get the state space matrices A,B
        representing (rough) discrete dynamics for a given true anomaly
        of the reference
        x_dot = A*x + B*u ->
        x_k1 = A_k*x_k + B_k*u_k

        where:  x_k = [x, y, z, x_dot, y_dot, z_dot].T
                u_k = [f_x, f_y, f_z].T (specific force)
                and x_dot = dx/dnu, is the derivative wrt true anomaly
                A_k = A*dnu+I, B_k = B*dnu

        if true anomaly is a vector, returns a list of A,B corresponding
            to the values of true anomaly
        '''
        n = self.orbit_ref.n
        ecc = self.orbit_ref.ecc
        one_e_cnu = 1+ecc*np.cos(nu)
        dnu_dt = n * (one_e_cnu)**2 / (1-ecc*ecc)**1.5
        R = self.orbit_ref.p/(one_e_cnu)
        
        if not hasattr(nu,'__len__'):
            A = np.zeros((6,6))
            A[0,3] = 1
            A[1,4] = 1
            A[2,5] = 1
            A[3,0] = 3/one_e_cnu
            A[3,4] = 2
            A[4,3] = -2
            A[5,2] = -1

            B = np.zeros((6,3))
            B[3,0] = 1/dnu_dt/R
            B[4,1] = 1/dnu_dt/R
            B[5,2] = 1/dnu_dt/R
            return A,B
        num = len(nu)
        A = np.zeros((6,6,num))
        A[0,3,:] = 1
        A[1,4,:] = 1
        A[2,5,:] = 1
        A[3,0,:] = 3/one_e_cnu
        A[3,4,:] = 2
        A[4,3,:] = -2
        A[5,2,:] = -1

        B = np.zeros((6,3,num))
        B[3,0,:] = 1/dnu_dt/R
        B[4,1,:] = 1/dnu_dt/R
        B[5,2,:] = 1/dnu_dt/R
        return A,B


    def posVelToState(self, pos, vel, nu):
        '''
        calculates the state in the TH system (at given true anomaly) 
        that corresponds to the given position and velocity.
        '''
        norm = np.linalg.norm
        cnu = math.cos(nu)
        snu = math.sin(nu)
        a = self.orbit_ref.a
        n = self.orbit_ref.n
        p = self.orbit_ref.p
        ecc = self.orbit_ref.ecc

        one_e_cnu = 1+ecc*cnu
        dnu_dt = n * (one_e_cnu)**2 / (1-ecc*ecc)**1.5

        E = np.arccos( (ecc+cnu)/one_e_cnu )
        if nu > math.pi:
            E = 2*math.pi - E
        cE = math.cos(E)
        sE = math.sin(E)

        R_ref = p/one_e_cnu
        # perifocal position on orbit corresponding to true anomaly
        ref_pos_pqw = R_ref*np.array([cnu,snu,0]).T
        ref_vel_pqw = a*n/(1-ecc*cE) * np.array([-sE, math.sqrt(1-ecc*ecc)*cE, 0]).T
        V_ref = norm(ref_vel_pqw)
        # given position and velocity expressed in perifocal
        dcm_eci_to_peri = self.orbit_ref.R_peri_to_eci.T
        pos_pqw = dcm_eci_to_peri @ pos
        vel_pqw = dcm_eci_to_peri @ vel
        
        # using the non-dimensional interpretation of the equations
        # x is the proportion of radius beyond R_ref in plane
        x = (norm(pos_pqw[0:2])-R_ref)/R_ref
        # this is incorrect but should be close for not extremely elliptical orbits
        x_pr = (norm(vel_pqw[0:2])-V_ref)/V_ref / dnu_dt
        x_pr = (norm(vel_pqw[0:2])-V_ref)/V_ref

        # y is the angle relative to true anomaly of the ref
        # true anomaly of pos
        nu_pos = self.orbit_ref.matchNu(pos)
        # we should make sure this is within +-pi for lienarization purposes
        y = nu_pos - nu
        y_pr = norm(vel_pqw[0:2])/norm(pos_pqw[0:2])/dnu_dt - 1

        # z is simply out of plane distance normalized by R_ref
        z = pos_pqw[2]/R_ref
        z_pr = vel_pqw[2]/V_ref / dnu_dt
        z_pr = vel_pqw[2]/V_ref
        
        state = np.array([x,y,z,x_pr,y_pr,z_pr])
        return state

def th_optimization(x_th_0, x_th_f, nu_0, nu_f, num_revs, th_ref):
    '''
    convex optimization of approximation of th equations

    Args:
        x_th_0: initial state in the th system
        x_th_f: final state in the th system
        nu_0: initial true anomaly
        nu_f: final true anomaly
        num_revs: number of revolutions, adds 2pi*n to anomaly sweep
        th_ref: reference th system

    Returns:
        T: time vector, 6xN+1
        X_T: matrix of state time history, 6xN+2
        U_T: matrix of control inputs, 6xN+1
    '''

    # create true anomaly steps
    N = 50 # number of discrete steps
    N = 500 # number of discrete steps
    start_true_anom = nu_0
    end_true_anom = nu_f + 2*math.pi*(num_revs + (nu_0>nu_f))
    true_anoms = np.linspace(start_true_anom, end_true_anom, N+1)
    dnu = (end_true_anom-start_true_anom) / N
    # calculate the time corresponding to each true anomaly
    T = th_ref.orbit_ref.trueAnomToTime(true_anoms)

    A,B = th_ref.state_space(true_anoms)

    # represents x_1 ... x_N
    X = cvx.Variable(6*(N))
    # u_0 ... u_N
    U_low_thrust = cvx.Variable(3*(N+1))
    U_high_thrust = cvx.Variable(3*(N+1))
    U = U_low_thrust + U_high_thrust
    # vectorize
    X_k  = cvx.hstack( (x_th_0, X) )
    X_k1 = cvx.hstack( (X, x_th_f) )
    # euler integration
    A_mat = block_diag(*np.transpose(A,[2,0,1])) * dnu + np.eye((N+1)*6)
    B_mat = block_diag(*np.transpose(B,[2,0,1])) * dnu

    # remove the final y constraint, which should allow the final 
    # value of true anomaly to float
    A_mat = np.delete(A_mat, -5, 0)
    B_mat = np.delete(B_mat, -5, 0)
    x_th_f_red = np.delete(x_th_f, -5)
    X_k1 = cvx.hstack( (X, x_th_f_red) )
    
    dyn = X_k1 == A_mat @ X_k + B_mat @ U
    # TODO: parameterize all the spacecraft parameters like these
    # low thrust vector constraint 
    ctrl_constr_low = cvx.norm(cvx.reshape(U_low_thrust, [3,N+1]), 2, 0) <= 0.01
    #ctrl_constr_low = cvx.norm(cvx.reshape(U_low_thrust, [3,N+1]), 2, 0) <= 1e-4
    # high thrust vector constraint 
    ctrl_constr_high = cvx.norm(cvx.reshape(U_high_thrust, [3,N+1]), 2, 0) <= 1
    # L-1 norm minimization is minimizing thrust assuming 
    # only axis aligned thrusting
    # L-2 norm minimization assumes rotation independent
    # represents relative isp efficiency of each
    obj = cvx.norm(U_low_thrust,1) + 50*cvx.norm(U_high_thrust,1)
    # optional additional regularization to promote high thrust impulsiveness
    obj_reg = 5*cvx.norm(U_high_thrust[3:]-U_high_thrust[:-3],1)
    obj_reg = 0

    constr = [dyn, ctrl_constr_low, ctrl_constr_high]

    prob = cvx.Problem(cvx.Minimize(obj+obj_reg), constr)

    prob.solve()
    status = prob.status
    print(status)
    if status == 'infeasible':
        return(-1, [],[],[],[],[])

    X_opt = X_k.value
    U_low_opt = U_low_thrust.value
    U_high_opt = U_high_thrust.value

    X_T = np.hstack((X_opt, x_th_f))

    X_T = X_T.reshape((-1,6)).T
    U_low_opt = U_low_opt.reshape((-1,3)).T
    U_high_opt = U_high_opt.reshape((-1,3)).T

    # calculate the required accelerations in the ECI frame
    # done here since we have true anomaly already
    orbit = copy.copy(sys.orbit_ref)
    accel_eci = np.zeros(U_low_opt.shape)
    for i in range(N+1):
        orbit.nu = true_anoms[i]
        r_ref, v_ref = orbit.toPosVel()
        # build dcm
        x_ref = r_ref / np.linalg.norm(r_ref)
        z_ref = np.cross(x_ref, v_ref)
        z_ref /= np.linalg.norm(z_ref)
        y_ref = np.cross(z_ref, x_ref)
        dcm_ref_to_eci = np.vstack([x_ref,y_ref,z_ref]).T
        accel_eci[:,i] = dcm_ref_to_eci @ ( U_low_opt[:,i] + U_high_opt[:,i] )
   
    return (0, T, X_T, U_low_opt, U_high_opt, accel_eci)

if __name__== '__main__':
    mu_earth = 398600.44
    #oe_initial = [ 0.95979589,  0.25192413,  0.09891915,  0.22551093, -0.06874523, 4.29373312]
    #oe_final = [ 0.96739787,  0.23921038,  0.07056417,  0.21252638, -0.06871709, 4.31422091]
    #oe_final = [ 1.1,  0.23921038,  0.07056417,  0.21252638, -0.06871709, 4.31422091]
    oe_gto = np.array([ 24.344e3, 0.5, 0.5, 0.1, 0.5, 0 ])
    oe_geo = np.array([ 42.164e3, 0.0, 0.0, 0.0, 0.0, 1 ])
    oe_initial = oe_gto
    # linear interpolation
    oe_final   = (oe_geo-oe_gto) * .05 + oe_gto
    oe_final   = (oe_geo-oe_gto) * .2 + oe_gto

    orbit = Orbit(oe_final,mu_earth,0)
    pos,vel=Orbit(oe_initial, mu_earth,0).toPosVel()
    nu = orbit.matchNu(pos)
    sys = TschaunerHempelSys(orbit)
    x_th_0 = sys.posVelToState(pos,vel,nu)
    x_th_f = np.array([0,0,0,0,0,0])
    nu_0 = 1
    nu_f = 6
    num_revs = 10
    num_revs = 1
    status, T, X_opt, U_low_opt, U_high_opt, accel_eci = th_optimization(
                                                x_th_0, x_th_f, nu_0, nu_f, 
                                                num_revs, sys)

    #import pdb; pdb.set_trace()
    if status == -1:
        print('no trajectory possible')
    else:
        plt.figure()
        plt.plot(X_opt[1,:-1], X_opt[0,:-1],label='traj')
        plt.plot(X_opt[1,::5], X_opt[0,::5],'.',label='equal spacing')
        plt.plot(x_th_0[1],x_th_0[0],'x',label='start')
        plt.plot(x_th_f[1],x_th_f[0],'x',label='target')
        plt.xlabel('y_cw')
        plt.ylabel('x_cw')
        plt.title('optimized trajectory')
        plt.gca().invert_xaxis()

        plt.figure()
        plt.plot(T, X_opt[2,:-1])
        plt.xlabel('T')
        plt.ylabel('z_cw')
        plt.figure()
        plt.plot(T, U_low_opt.T)
        plt.legend(['f_low_x','f_low_y','f_low_z'])
        plt.figure()
        plt.plot(T, U_high_opt.T)
        plt.legend(['f_high_x','f_high_y','f_high_z'])

        fig, axs = plt.subplots(2,3)

        axs[0,0].plot(T, X_opt[0,:-1])
        axs[0,0].set_xlabel('T')
        axs[0,0].set_ylabel('x_th')

        axs[0,1].plot(T, X_opt[1,:-1])
        axs[0,1].set_xlabel('T')
        axs[0,1].set_ylabel('y_th')

        axs[0,2].plot(T, X_opt[2,:-1])
        axs[0,2].set_xlabel('T')
        axs[0,2].set_ylabel('z_th')

        axs[1,0].plot(T, X_opt[3,:-1])
        axs[1,0].set_xlabel('T')
        axs[1,0].set_ylabel('x_th_pr')

        axs[1,1].plot(T, X_opt[4,:-1])
        axs[1,1].set_xlabel('T')
        axs[1,1].set_ylabel('y_th_pr')

        axs[1,2].plot(T, X_opt[5,:-1])
        axs[1,2].set_xlabel('T')
        axs[1,2].set_ylabel('z_th_pr')

        plt.show()
