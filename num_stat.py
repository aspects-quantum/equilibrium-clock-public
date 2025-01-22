import numpy as np
from functools import reduce

from scipy.optimize import curve_fit
from scipy.stats import expon

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8

})

col_scheme_BR = ['midnightblue','indigo','mediumvioletred','maroon','darkorange','gold','yellow','olive','darkgreen','teal',"midnightblue","mediumblue","green","seagreen","firebrick","darkred","darkorange","burlywood","purple","magenta",]

# Helper function to find subsequences in arrays
# from: https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
def np_search_sequence(a, seq, distance=1):
    return np.where(reduce(lambda a,b:a & b, ((np.concatenate([(a == s)[i * distance:], np.zeros(i * distance, dtype=np.uint8)],dtype=np.uint8)) for i,s in enumerate(seq))))[0]

def rate_err(sample, M = 50, Nprime_range = (10**np.linspace(1,2,25)).astype(int)):

    N = len(sample)

    EG_range   = np.zeros((len(Nprime_range)))
    VarG_range = np.zeros((len(Nprime_range)))

    ET_range   = np.zeros((len(Nprime_range)))
    VarT_range = np.zeros((len(Nprime_range)))

    T0 = min(sample)

    AVG = np.average(sample) - np.min(sample)

    loc, labda = expon.fit(sample)

    HatGamma = 1/labda

    for Nk, Nprime in enumerate(Nprime_range):    
        G_sample = np.zeros((M))
        T_sample = np.zeros((M))

        for Mk in range(M):
            idx_random = np.random.permutation(np.arange(N))[:Nprime]

            sample_slice = sample[idx_random]

            loc_, labda_ = expon.fit(sample_slice)
            G_sample[Mk] = 1/labda_
            T_sample[Mk] = np.average(sample_slice)

        EG_range[Nk] = np.average(G_sample)
        VarG_range[Nk] = np.sum((G_sample-HatGamma)**2)/(M-1) # np.var(G_sample,ddof=1)

        ET_range[Nk] = np.average(T_sample)
        VarT_range[Nk] = np.var(T_sample,ddof=1)


    z = np.polyfit(np.log(Nprime_range),np.log(VarG_range*Nprime_range),0)

    y = np.polyfit(np.log(Nprime_range),np.log(VarT_range*Nprime_range),0)

    intrinsic_error_relative = np.sqrt(1/2 * (np.exp(y[0])-(AVG)**2)) / AVG


    ###############################
    # Visualization for Bootstrap #
    # method. Set to "True" for   #
    # plotting.                   #
    ###############################
    if False:
        fig, axs = plt.subplots(1,2,figsize=(7.08333,2.0))

        axs[0].errorbar(Nprime_range,VarT_range,yerr=VarT_range/np.sqrt((M-1)/2),capsize=3,
                        color=col_scheme_BR[0],marker=".",linestyle="",label=r"${\rm Var}_{M,N'}[\widehat T]$")
        axs[0].loglog(Nprime_range,np.exp(y[0])/Nprime_range,color=col_scheme_BR[2],label=r"fit $(\widehat T ^2 + 2\eta^2) / N'$")
        axs[0].loglog(Nprime_range,AVG**2/Nprime_range,linestyle="--",color="black",label=r"$\widehat T ^2 / N'$")
        axs[0].legend(loc="upper right")
        axs[0].set_ylabel(r"${\rm Var}[\widehat T]$ (s$^2$)")
        axs[0].set_xlabel(r"subsample size $N'$")
        axs[0].text(-0.1, 0.94, "(a)",transform=axs[0].transAxes)


        axs[1].errorbar(Nprime_range,VarG_range,yerr=VarG_range/np.sqrt((M-1)/2),capsize=3,
                        color=col_scheme_BR[0],marker=".",linestyle="",label=r"${\rm Var}_{M,N'}[\widehat\Gamma]$")
        axs[1].loglog(Nprime_range,np.exp(z[0])/Nprime_range,color=col_scheme_BR[2],label=r"fit $\widehat \alpha \Gamma ^2 / N'$")
        axs[1].loglog(Nprime_range,HatGamma**2/Nprime_range,linestyle="--",color="black",label=r"$\widehat \Gamma ^2 / N'$")
        axs[1].legend(loc="upper right")
        axs[1].set_ylabel(r"${\rm Var}[\widehat\Gamma]$ (s$^{-2}$)")
        axs[1].set_xlabel(r"subsample size $N'$")
        axs[1].text(-0.1, 0.94, "(b)",transform=axs[1].transAxes)
        
        plt.tight_layout()
        plt.savefig("figs/paper/bootstrap.svg")#,dpi=600)
        plt.show()

    return np.sqrt(np.exp(z[0]) / N + HatGamma*intrinsic_error_relative) # np.sqrt((1/labda)**2 / len(sample) + eta2 * (1/labda)**4)


class RateStatistics:
    # Class to convert discretized three-level trace into estimates for stochastic rates
    def __init__(self,states,time,identifier):
        """
        Generate arrays of jump times between all possible level-combinations
        Input:
            states  :   array of length N with values 0,1,2
            time    :   array of length N with time corresponding to states
        """
        self.states = states
        self.time = time
        self.identifier = identifier

        pass

    def time_stats_conditional(self):
        """
        Obtain the conditional rates

        plot    :   if True, generate plots
        rates   :   if True, return the following

        M, GammaCond, GammaMarkov

        where M = [[-\Gamma_0,    \Gamma_{01}, \Gamma_{02}],
                   [ \Gamma_{10},-\Gamma_1,    \Gamma_{12}],
                   [ \Gamma_{20}, \Gamma_{21},-\Gamma_2   ]]
            is the rate matrix, note that: \Gamma_0 = \Gamma_{10} + \Gamma_{20} and similar for the other columns!

        where GammaCond = [[10,20],[21,01],[02,12]], with ij denoting the average rate at which the system jumps from
            j to i (j->i) pre-conditioned on this type of jump. So, this rate is not the same as the entries in M, but
            they should crucially be the same for 10 and 20 and likewise for the rest

        where GammaMarkov = [[[10|1, 10|2], [20|1, 20|2]],
                             [[21|2, 21|0], [01|2, 01|0]],
                             [[02|0, 02|1], [12|0, 12|1]]]
            are the conditional exiting rates, i.e., like the ones above but now pre-conditioned on the sequence i->j->k.
            Ideally, the last of the three indices does not matter
        """
        # Possible transitions
        # Convention: [a,b,c] = a<-b|c
        #                              j = 0             j = 1
        #                          k = 0   k = 1     k = 0   k = 1
        transition =  np.array([[[[2,0,1],[2,0,2]],[[1,0,1],[1,0,2]]],      # i = 0
                                [[[0,1,0],[0,1,2]],[[2,1,0],[2,1,2]]],      # i = 1
                                [[[1,2,1],[1,2,0]],[[0,2,1],[0,2,0]]]])     # i = 2

        # Index convention
        # transition[i,j,k] : i-th column, j-th row, k: 

        time_stats = (self.get_time_stats(transition.reshape((12,3))))
        
        # Index reshaper (compatible with the above convention!)
        # note: transition[i,j,k] = j+i+2 <- i |
        indexer = lambda i,j,k : k + 2*j + 4*i

        # Generate return matrices
        M = np.zeros((3,3))
        GammaCond = np.zeros((3,2))
        GammaMarkov = np.zeros((3,3,2))

        M_err = np.zeros((3,3))

        dt = self.time[1]-self.time[0]

        # Reshape time_stats
        for i in range(3):      # COLUMNS
            # Determine the survival rate
            stats_survival = np.concatenate((time_stats[indexer(i,0,0)],time_stats[indexer(i,0,1)],
                                             time_stats[indexer(i,1,0)],time_stats[indexer(i,1,1)]))
            avg_survival = np.average(stats_survival)

            # Fitting
            loc, labda = expon.fit(stats_survival,scale=1/avg_survival)

            # Determine exiting rates
            n1 = len(time_stats[indexer(i,0,0)]) + len(time_stats[indexer(i,0,1)])
            n2 = len(time_stats[indexer(i,1,0)]) + len(time_stats[indexer(i,1,1)])
            weights = np.array([n1,n2])/(n1+n2)

            leaving_rate_i = 1/labda
            transition_rate_i = 1/labda * weights

            # Error using bootstrap technique
            N = len(stats_survival)
            M_bootstrap = 100
            N_samples = 200

            gamma_bootstrap = np.zeros((N_samples))

            for Nk in range(N_samples):
                idx_random = np.random.randint(0,N,size=M_bootstrap)
                stats_survival_slice = stats_survival[idx_random]

                loc, labda = expon.fit(stats_survival_slice,scale=1/avg_survival)
                gamma_bootstrap[Nk] = 1/labda

            leaving_err1 = np.sqrt(np.var(gamma_bootstrap,ddof=1))

            #### TEMP
            leaving_err = rate_err(stats_survival)

            transition_err = weights * leaving_err

            # Add matrix entry for rate matrix
            M[i,i] = -leaving_rate_i
            M[(i-1)%3,i] = transition_rate_i[0]
            M[(i+1)%3,i] = transition_rate_i[1]

            # Add errors
            M_err[i,i] = leaving_err
            M_err[(i-1)%3,i] = transition_err[0]
            M_err[(i+1)%3,i] = transition_err[1]

            for j in range(2):  # ROW
                stats_k1 = time_stats[indexer(i,j,0)]
                stats_k2 = time_stats[indexer(i,j,1)]

                # List with both statistics
                stats_k = [stats_k1,stats_k2]

                # Combined statistics
                stats_combo = np.concatenate((stats_k1,stats_k2))
                avg_combo = 1/np.average(stats_combo)

                # Estimate unconditional rate
                loc, labda = expon.fit(stats_combo,scale=1/avg_combo)
                gamma_combo = 1/labda
                err_combo = gamma_combo / np.sqrt(len(stats_combo)) # gamma_combo**2 * np.std(stats_combo) / np.sqrt(len(stats_combo))

                # Add to return array,
                # index has to be separately treated 
                GammaCond[i,(j+1)%2] = gamma_combo

                # Obtain the histogram PDF
                PDFs, bin_mids = [], []
                gamma_fit = []
                errors = []
                locs = []
                avg = []
                var = []

                for stats in stats_k:
                    # Obtain data histogram
                    temp1, temp2 = np.histogram(stats,20,density=True)
                    temp3 = temp2[:-1] + 0.5 * (temp2[1]-temp2[0]) # midpoints
                    PDFs.append(temp1)
                    bin_mids.append(temp3)
                    avg.append(np.average(stats))
                    var.append(np.var(stats))

                    errors.append(np.sqrt(len(stats))/np.sum(stats))

                    # Maximum likelyhood estimation of the rate parameter
                    if len(stats) > 1:
                        loc, labda = expon.fit(stats,scale=1/avg[-1])
                        gamma_fit.append(1/labda)
                        locs.append(loc)
                    else:
                        gamma_fit.append(1/avg[-1])
                        locs.append(0.0)

                # Add Markov check rates to return statement
                if i==0:
                    GammaMarkov[i,(j+1)%2,0] = gamma_fit[0]
                    GammaMarkov[i,(j+1)%2,1] = gamma_fit[1]
                elif i==1:
                    GammaMarkov[i,(j+1)%2,0] = gamma_fit[1]
                    GammaMarkov[i,(j+1)%2,1] = gamma_fit[0]
                elif i==2:
                    GammaMarkov[i,(j+1)%2,0] = gamma_fit[1]
                    GammaMarkov[i,(j+1)%2,1] = gamma_fit[0]

        return M, GammaCond, GammaMarkov, M_err

    def get_time_stats(self,transition,print_debug=False):
        """
        Generate array of the measured times for the transitions
            transition  :   array [j,i] or array [j,i,k]
                            or two-dim array with axis 0 adressing the transition
                            and axis 1 are elements as [i,j] ..
        return statistics for W[t,j|i] or W[t,j|i,k]
        """
        # Make sure transition is a 2d array
        transition = np.atleast_2d(transition)

        # Obtain the index of the state right before the jump
        # i.e. [0,0,1] would give back idx [1]
        jumps = np.concatenate((np.diff(self.states),np.array([0.])))
        jumps_idx = np.where(jumps!=0)[0]

        # Obtain subset of states and times when the jumps happened
        # i.e. states_collapsed[k] would be the state from where the jump happened
        #      and times_collapsed[k] would be the time when the jump happened
        states_collapsed = self.states[jumps_idx]   
        times_collapsed = self.time[jumps_idx]

        # Return list
        time_stats = []

        if transition.shape[1] == 2:
            # Case where we look at unconditional transitions
            for k in range(transition.shape[0]):
                # Cycle through all possible transitions
                trans_k = transition[k,:]
                trans_k = trans_k[::-1]

                # Get indices of jump start
                idx = np_search_sequence(states_collapsed,trans_k)

                # Note that here, the index must be shifted because
                # states_collapsed[k] is the the state from where the jump happened
                # thus times_collapsed[idx] is the time of the jump trans_k,
                # and times_collapsed[idx-1] is the time of the previous jump
                time_stats_k = times_collapsed[idx]-np.concatenate(([0.0],times_collapsed))[idx]

                time_stats.append(time_stats_k)

                if print_debug:
                    print("SEQUENCE :",transition[k,:])
                    print("NUM SAMP :",len(time_stats[-1]))
                    print("AVG TIME :",np.average(time_stats[-1]))
                    print("DEV TIME :",np.sqrt(np.var(time_stats[-1])))

        elif transition.shape[1] == 3:
            # Case where we look at conditional transitions
            for k in range(transition.shape[0]):
                # Cycle through all possible transitions
                trans_k = transition[k,:] # given as [i,j,k] to mean
                                          # jump j<-i given k
                trans_k = trans_k[::-1]   # flip to [k,j,i] chronologically

                # Get indices of jump destination
                idx = np_search_sequence(states_collapsed,trans_k)

                time_stats_k = times_collapsed[idx+1]-times_collapsed[idx]

                time_stats.append(time_stats_k)

                if print_debug:
                    print("SEQUENCE : "+str(trans_k[2])+" <- "+str(trans_k[1])+" | "+str(trans_k[0]))
                    print("NUM SAMP :",len(time_stats[-1]))
                    print("AVG TIME :",np.average(time_stats[-1]))
                    print("DEV TIME :",np.sqrt(np.var(time_stats[-1])))

        else:
            raise Exception("Problem with get_time_stats(), length of the transition array must be 2 or 3.")

        return time_stats

class NumberStatistics:
    # Class to convert tick sequence into statistics for Number

    def __init__(self,rl_times,lr_times,method):
        """
        Generates array NT of shape (2,n_ticks) with first row containing tick number
        and the second row containing time tags
        Input:  
            rl_times : times of right-left transitions
            lr_times : times of left-right transitions
            method   : "bi" for bidirectional, "uni" for unidirectional
        """
        self.rl_times = rl_times
        self.lr_times = lr_times
        self.method = method

        if method == "bi":
            self.NT = self.generate_NT_bi()
        elif method == "uni-rl":
            self.NT = self.generate_NT_uni(direction="rl")
        elif method == "uni-lr":
            self.NT = self.generate_NT_uni(direction="lr")
        else:
            raise Exception("method must be uni or bi","eggs")
        pass

    def generate_NT_uni(self,direction="rl"):
        """
        Method to generate the NT array for unidirectional transitions
        """
        if direction == "rl":
            sign=1.0
            times = self.rl_times
        elif direction == "lr":
            sign=-1.0
            times = self.lr_times
        else:
            raise Exception('direction must be rl or lr')
        
        n_ticks = len(times)
        return np.hstack((np.array([[0.0],[0]]),np.vstack((sign*np.arange(1,n_ticks+1),times))))

    def generate_NT_bi(self):
        """
        Method to generate the NT array for bidirectional transitions
        Input:
            rl_times
        """
        n_ticks = len(self.rl_times) + len(self.lr_times)

        # Prepare the NT array as increments in first row, times in second row
        NT = np.zeros((2,n_ticks))
        NT[1,:len(self.rl_times)] = self.rl_times
        NT[0,:len(self.rl_times)] = 1.0
        NT[1,len(self.rl_times):] = self.lr_times
        NT[0,len(self.rl_times):] = -1.0

        # Order array according to ascenting time tags
        tick_order = np.argsort(NT[1,:])
        NT = NT[:,tick_order]

        # Transform tick increments into tick number
        NT[0,:] = np.cumsum(NT[0,:])

        # Add zero-th tick
        NT = np.hstack((np.array([[0.0],[0]]),NT))

        return NT

    def N_of_t(self,times):
        """
        Vectorized function to return an array of N evaluated at the supplied times
            times : 1-dim array
        Returns array of the same shape as times
        """
        # Ensure vector format for times
        times = np.atleast_1d(times)

        # Read out numbers at times
        N_eval = np.zeros_like(times)

        # Iterate over all tick events
        for k, tk in enumerate(self.NT[1,1:]):
            N_eval[np.where(((times<tk)*(times>self.NT[1,k])))] = self.NT[0,k]

        return N_eval
    
    def get_samples(self,times,mode="STD"):
        """
        Vectorized function returns statistical as many trajectory samples of N(times)
        as possible with the given measurement record
            mode    :   "STD" -- standard mode cuts apart time trace into intervals
                        "IID" -- shift the array such that it always starts right after a jump
        """
        # Determine possible sample size
        sample_size = int(np.floor((self.NT[1,-1])/np.max(times)))
        if sample_size < 1000:
            print("Warning, your sample-size isn't sufficiently large")
            print("for a significant analysis. Recommended sample-size")
            print("is 1000, you have "+str(sample_size))

        # Generate samples on which to evaluate statistics
        N_samples = np.zeros((sample_size,len(times)))
        if mode == "IID":
            sample_size-= 10
            idx = np.argmax(self.rl_times>=0)
        for k in range(sample_size):
            if mode == "STD":
                temp = self.N_of_t(times[-1]*k + times)
                N_samples[k,:] = temp - temp[0]
            elif mode == "IID":
                # Obtain N statistics
                # temp = self.N_of_t(self.NT[1,idx]+1e-9 + times)
                temp = self.N_of_t(self.rl_times[idx]+1e-9 + times)
                N_samples[k,:] = temp - temp[0]

                # Set next index
                # idx = np.argmax(self.NT[1,:]>=times[-1]*(k+1))
                idx = np.argmax(self.rl_times>=times[-1]*(k+1))

        return N_samples

    def precision(self,times,errorbar=False,estimator=True):
        """
        Vectorized function returns statistical expectation value for the
        generalized clock precision.
        Make sure that the max(times) is at least 1000 smaller than the total
        """
        sample = self.get_samples(times)

        avg = np.average(sample,axis=0)
        var = np.var(sample,axis=0)

        # Temporary fix for vanishing variance ...
        var[np.where(var==0)] = 1e-3
        avg[np.where(avg==0)] = 1e-3

        # Generate error bar
        if estimator:
            if errorbar:
                S = np.abs(avg**2/var)/times
                m = len(sample)
                err = S / np.sqrt(m) * np.sqrt(4 * var / avg**2 + 2*m/(m-1))
                return S, err
            else:
                return np.abs(avg**2/var)/times

        else:
            if errorbar:
                m = len(sample)
                err = np.sqrt(1/(m*var) + 2/(m-1) * avg**2 / var**2)
                return np.abs(avg/var), err 

            return np.abs(avg/var)
    

class TimeEstimator:
    # Superclass for estimating times
    def __init__(self,times,states,M):
        self.times = times
        self.states = states
        self.M = M
        pass

    def getSlicy(self,m):
        """
        Return sequence of sliced states
        """
        length_full = len(self.states)
        length_slice = int(length_full/m)
        return np.reshape(self.states[:m*length_slice],(m,length_slice)), self.times[length_slice]-self.times[0]
    
    def getStats(self,slicy):
        """
        Return how often each state appears in each slice
        Return format
        jump_stats.shape = (m,3)
        """
        slicy = np.atleast_2d(slicy)

        # Get array of jump indices
        jumpy_jump = np.diff(slicy,axis=1)
        # Original shape
        shapy = np.shape(slicy)

        # Resulting array
        res = np.zeros((shapy[0],3),dtype=int)

        # Cycle through all slices
        for k in range(shapy[0]):
            # Index of jumps
            jumpy_idx = np.where(jumpy_jump[k,:]!=0)
            # Get states from where jump occurred
            states = (slicy[k,:])[jumpy_idx]
            # Add final state too!
            states = np.concatenate((states,[slicy[k,-1]]))

            # Return the counts of each state
            # DANGER: if slice too small and not all three states appear
            # this function fails.
            temp_state, temp_counts = np.unique(states,return_counts=True)
            if len(temp_state)==3:
                res[k,:] = temp_counts
            else:
                if temp_state[0]!=0:
                    temp_state = np.concatenate(([0],temp_state))
                    temp_counts = np.concatenate(([0],temp_counts))
                if temp_state[1]!=1:
                    temp_state = np.insert(temp_state,1,1)
                    temp_counts = np.insert(temp_counts,1,0)
                if len(temp_state)==2:
                    temp_state = np.concatenate((temp_state,[2]))
                    temp_counts = np.concatenate((temp_counts,[0]))

                res[k,:] = temp_counts

        return res
    
    def getEstimator(self,counts):
        """
        Return the time estimator based on input counts
        Input:  
            counts  :   array of shape (3) or (m,3)
        Output:
            array of shape (1,) or (m,)
        """
        # Prep input array
        counts = np.atleast_2d(counts)

        # Dead time
        dt = self.times[1]-self.times[0]

        # Get times
        # Add time increment from dead time
        diag_times = np.abs(1/np.diagonal(self.M)) + dt

        # Return estimator
        return counts.dot(diag_times) 


    def getLateSNR(self,m,errorbar=False):
        """
        Return the SNR for the time estimator
        Input:
            m           :   number of slices, should ideally be ~100 - ~1000
            errorbar    :   True return errorbar, false not
        """
        # Slice array into m subarrays
        slicy, tf = self.getSlicy(m)
        # Get counting stats
        counts = self.getStats(slicy)
        # Get estimator samples
        estimators =  self.getEstimator(counts)

        # Average etc.
        avg = np.average(estimators)
        var = np.var(estimators)

        if errorbar:
            S = np.abs(avg**2/var) / tf
            err = S / np.sqrt(m) * np.sqrt(4 * var / avg**2 + 2*m/(m-1))
            return S, err 
        
        return avg**2 / var