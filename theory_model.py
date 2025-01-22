import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import null_space

from num_stat import NumberStatistics

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8

})

class DQDClock:
    '''
    Class defining the theoretic double quantum dot (DQD) clock
    '''
    def __init__(self,omega,coupling_baths,beta_baths,mu_baths,kappa):
        self.omega = omega
        self.kappa = kappa
        self.beta_baths = beta_baths
        self.coupling_baths = coupling_baths
        self.mu_baths = mu_baths
        self.p0 = np.array([1.0,0,0])

        self.set_gammas()
        self.set_M()

        pass

    def set_gammas(self):
        """
        Function to define the effective jump rates between bath and double dot
        """
        def n(beta,mu):
            return 1/(np.exp(beta*(self.omega - mu))+1)
        
        self.gammaL      = n(self.beta_baths[0],self.mu_baths[0])*self.coupling_baths[0]
        self.gammaLprime = (1-n(self.beta_baths[0],self.mu_baths[0]))*self.coupling_baths[0]

        self.gammaR      = (1-n(self.beta_baths[1],self.mu_baths[1]))*self.coupling_baths[1]
        self.gammaRprime = n(self.beta_baths[1],self.mu_baths[1])*self.coupling_baths[1]

        pass

    def set_M(self):
        M = np.array([[-self.gammaL - self.gammaRprime,self.gammaLprime,self.gammaR],
                      [self.gammaL,-self.kappa - self.gammaLprime, self.kappa],
                      [self.gammaRprime,self.kappa,-self.kappa-self.gammaR]],dtype=np.complex128)
        self.M = M
        
        pass

    def Mmod(self,chi,scheme=0):
        '''
        pre-defined couting schemes are
            0   :   anti-symmetric counting of current tunneling on the right dot
            1   :   uni-directional counting of electron jumps onto right dot
            2   :   all jumps count as ticks
        '''
        _M = self.M.copy()
        if scheme==0:
            _M[0,2]*=np.exp(1j*chi)
            _M[2,0]*=np.exp(-1j*chi)
            return _M
        if scheme==1:
            _M[0,2]*=np.exp(1j*chi)
            return _M
        if scheme==2:
            for i in range(3):
                for j in range(i):
                    _M[i,j]*= np.exp(1j*chi)
                    _M[j,i]*= np.exp(1j*chi)
            return _M
        
    def getAccuracy(self,scheme):
        def maxEval(chi):
            eigenvalues = np.linalg.eigvals(self.Mmod(chi,scheme))
            return eigenvalues[np.argmax(np.real(eigenvalues))]

        def avgN():
            # Determine average number
            return -1j*ddx(maxEval,0)

        def varN():
            # Determine number variance
            return -ddx2(maxEval,0)

        return np.abs(np.real(avgN()/varN()))
    
    def getSteadyState(self):
        v0 = null_space(self.M)[:,0]
        v0 = v0/np.sum(v0)
        return v0


    def stochasticTrajectory(self,times):
        dt = times[1]-times[0]
        p_res = np.zeros((len(times),3))
        p_res[0,:] = self.p0

        for k,t in enumerate(times[:-1]):
            prob_temp = p_res[k,:] + dt*self.M.dot(p_res[k,:])
            choice = np.random.choice(3,1,p=np.real(prob_temp))
            p_res[k+1,choice] = 1.0

        lvl_list = p_res.dot(np.array([0,1,2]))
        lr_times = []
        rl_times = []

        for k,t in enumerate(times[:-1]):
            if np.allclose(lvl_list[k],2.0) and np.allclose(lvl_list[k+1],0.0):
                lr_times.append(t)
            if np.allclose(lvl_list[k],0.0) and np.allclose(lvl_list[k+1],2.0):
                rl_times.append(t)
        
        return times,p_res,np.array(lr_times),np.array(rl_times)
        
def ddx(f,x,diff = 1e-2):
    # Numerical first order derivative
    return (f(x+diff/2)-f(x-diff/2))/diff

def ddx2(f,x,diff = 1e-2):
    # Numerical second order derivative
    return (f(x+diff)-2*f(x)+f(x-diff))/diff**2

    