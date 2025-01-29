import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from scipy import signal

from scipy.special import erf

from data_loader import load_id

from num_stat import NumberStatistics

from reflecto_pwr import get_reflectometry_power

class ReadoutGaussian:
    """
    Gaussian PDF class
    """
    def __init__(self, mean=0, sigma=1, weight=1):
        self.mean = mean
        self.sigma = sigma
        self.amplitude = weight / (np.sqrt(2*np.pi*sigma**2))
        pass

    def __call__(self,x):
        # Return probability amplitude of at x.
        return self.amplitude * np.exp(-0.5*(x-self.mean)**2/self.sigma**2)
    
class ReadoutStates:
    """
    Class to convert current into discrete states
    """
    def __init__(self, sensor_data, nstate=3, nbins = 40, plot=True, ksmooth=2, dynamical_calib=False, gauss=True, **kwargs):
        self.sensor_data = sensor_data
        self.dynamical_calib = dynamical_calib
        if self.dynamical_calib:
            self.window = kwargs.get('window',1000)
            self.step = kwargs.get('step',100) 
        else:
            self.window = len(self.sensor_data)
            self.step = len(self.sensor_data)
        self.nstate = nstate
        self.plot = plot
        self.ksmooth = ksmooth
        self.calib_success = True
        self.gauss = gauss
        self.nbins = nbins
        pass

    def __call__(self):
        # Convert sensor current data into discrete signal

        # Collapse probabilities
        pstate = np.sum(self.state_class,axis=2)


        state = np.argmax(np.array(pstate), axis=1)

        # Determine error probability
        err = 1 - np.max(pstate,axis=1) / np.sum(pstate,axis=1)

        # State smoothing
        for i in range(1,len(state)-self.ksmooth):
            if state[i] != state[i-1] and state[i]==1:
                if not np.allclose(np.sum(np.abs(state[i:i+self.ksmooth]-state[i])),0):
                    # If not equal previous and next two states, set equal to previous state
                    state[i] = state[i-1]

        if self.plot:
            plt.figure()
            plt.plot(state*(self.sensor_data.max()-self.sensor_data.min())/3.0,marker=".",linestyle=None,color="red")
            plt.plot(self.sensor_data-self.sensor_data.min(),color="blue",linewidth=0.3)
            if self.dynamical_calib:
                plt.plot(np.linspace(0,len(self.sensor_data),self.n_iter),self.popt_all[:,0],label="mean low")
                plt.plot(np.linspace(0,len(self.sensor_data),self.n_iter),self.popt_all[:,3],label="mean mid")
                plt.plot(np.linspace(0,len(self.sensor_data),self.n_iter),self.popt_all[:,6],label="mean high")
                plt.legend(loc="best")
            plt.show()

        return state, err

    def calibrate(self,mode=0,directory=None,noisy=False,channel="B-V-"):
        """
        Calibration of data identifier
            calibration data : current traces
            mode:   0   generate opt param guess freshly
                    1   generate and WRITE opt param guess
                    2   READ opt parameter guess
            directory: has to be string if mode is 1 or 2
        """
        # Assert that directory is supplied in case of mode 1 or 2 usage
        assert not(mode>0 and directory == None)

        # Generate empty calibration data points
        self.n_iter = int((len(self.sensor_data)-self.window)/self.step)+1

        # Initialize optimal parameter and success flags
        self.popt_all = np.zeros((self.n_iter,self.nstate*3))
        succ_all = np.zeros((self.n_iter))

        # Classifiers
        self.state_class = np.zeros((len(self.sensor_data),self.nstate,int(self.window/self.step)))

        # Cycle through all steps
        for k in range(self.n_iter):
            # Data slice
            slice_k = self.sensor_data[k*self.step:k*self.step + self.window]

            # Histogram data for readout trace (squeeze along time axis)
            hist_PDF, bin_edges = np.histogram(slice_k, bins=self.nbins, range=[slice_k.min(), slice_k.max()], density=True)
            hist_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Sort data in ascending way
            data_sorted = np.sort(slice_k)
            cumulative_data = np.linspace(0,1,len(data_sorted))

            # Min and maximal current values
            i_min, i_max = data_sorted[0], data_sorted[-1]
            i_width = i_max - i_min

            if mode < 2:

                if noisy:
                    # In case of noisy data where the below method fails, proceed as follows
                    # 1. fit single gaussian
                    # 2. take two guess side-peaks

                    # Init param guess: mu, sigma, amplitude
                    q0 = [0.5*(i_min + i_max),i_width/10.0,np.max(hist_PDF)/(np.sqrt(2*np.pi))]

                    try:
                        qopt, qcov = curve_fit(gauss, hist_midpoints, hist_PDF, q0)
                        succ_all[k] = True
                    except:
                        print("An optimization error occurred:", error)                     
                        qopt = q0
                        succ_all[k] = False

                    # Make sigma positive
                    qopt[1] = np.abs(qopt[1])

                    # Set guesses
                    amplitudes = [qopt[-1]/5,qopt[-1]/5,qopt[-1]/5]
                    sigmas     = [3*qopt[1]/2,3*qopt[1]/2,3*qopt[1]/2]
                    means      = [qopt[0]-qopt[1],qopt[0],qopt[0]+qopt[1]]

                    # Combine
                    popt = [[means[i], sigmas[i], amplitudes[i]] for i in range(self.nstate)]
                    popt = np.array(popt).flatten()

                    if self.plot:
                        plt.figure()
                        plt.plot(hist_midpoints, hist_PDF, 'b.',label="measurement histogram")
                        plt.plot(hist_midpoints,gauss(hist_midpoints,*qopt),label="single gauss")
                        plt.plot(hist_midpoints,triple_point_readout(hist_midpoints,*popt),label="manual triple")
                        plt.vlines(popt[0::3],ymin=0,ymax=np.max(hist_PDF),label="opt means")
                        plt.legend(loc='best')
                        plt.show()

                else:
                    # Define initial guesses for classifier parameters
                    if k>=0:
                        # Define initial guesses for classifier parameters
                        amplitudes = [np.mean(hist_PDF) for _ in range(self.nstate)]
                        # amplitudes = [1/3,1/3,1/3]
                        sigmas = [0.05 * i_width for _ in range(self.nstate)]
                        means = np.linspace(i_min, i_max, self.nstate+12)[[2,6,9]]

                        p0 = [[means[i], sigmas[i], amplitudes[i]] for i in range(self.nstate)]
                        p0 = np.array(p0).flatten()

                    # Optimise means and standard deviations of readout Gaussians
                    try:
                        if channel=="D-V-":
                            popt, pcov = curve_fit(triple_point_readout, hist_midpoints, hist_PDF**2, p0=p0)
                        else:
                            bounds = ([i_min,0.0,0.0,i_min,0.0,0.0,i_min,0.0,0.0],[i_max,i_max,np.inf,i_max,i_max,np.inf,i_max,i_max,np.inf])

                            popt, pcov = curve_fit(triple_point_readout, hist_midpoints, hist_PDF**2, p0=p0,bounds=bounds)

                        succ_all[k] = True
                    except Exception as error:
                        print("An optimization error occurred:", error)                     
                        popt = p0
                        succ_all[k] = False

                    # Ensure all sigmas are positive
                    for i in range(self.nstate):
                        popt[i*self.nstate + 1] = np.abs(popt[i*self.nstate + 1])

                    # Ensure correct ordering of maxima
                    sort_idx = np.argsort([popt[i*self.nstate] for i in range(self.nstate)])
                    popt = [[popt[i*self.nstate],popt[i*self.nstate+1],popt[i*self.nstate+2]] for i in sort_idx]
                    popt = np.array(popt).flatten()

                self.popt_all[k,:] = popt

                if mode == 1:
                    np.save(directory+'/'+channel+'p0.npy',popt)
        
            elif mode == 2:
                popt = np.load(directory+'/'+channel+'p0.npy')
                p0=popt

            # Temporary amplitude normalizer
            _amp = np.sum(popt[2::3])
            _popt = popt.copy()
            _popt[2::3]/=_amp

            if self.plot and not noisy:
                plt.figure(figsize=(12, 5))
                plt.plot(hist_midpoints, hist_PDF, 'b.',label="measurement histogram")
                plt.plot(hist_midpoints, np.sqrt(triple_point_readout(hist_midpoints,popt)), 'r', label="gaussian fit")
                plt.vlines([p0[i*3] for i in range(3)],ymin=0,ymax=np.max(hist_PDF),colors='g',label="initial guess")
                plt.vlines([popt[i*3] for i in range(3)],ymin=0,ymax=np.max(hist_PDF),colors='orange',label="optimized")
                plt.legend(loc='best')
                plt.show()

            # Iterate over the three possible current states
            for i in range(self.nstate):
                # Optain the WEIGHTED gaussian probabilities
                class_temp = gauss(slice_k,_popt[i*3:(i+1)*3])

                # Correctly associate the probabilities
                #    0 - step   : -1
                # step - 2 step : -2
                # ...
                for j in range(int(self.window/self.step)):
                    self.state_class[(k+j)*self.step:(k+j+1)*self.step,i,-j] = class_temp[j*self.step:(j+1)*self.step]

class TickCounter:
    """
    Class to convert discrete state sequence into ticks for given
    method of defining ticks.
    """
    def __init__(self,tick=[0,1,2,0], plot=True, details=True):

        assert np.sum(np.diff(tick)) == 0
        self.tick = np.array(tick)
        self.plot = plot
        self.details = details
        pass

    def __call__(self, states, time):
        # Returns basic tick time statistics for given state sequence

        # Preprocess states to accurately count jumps across two states
        # (integration time captures intermediate state)
        # states_og = np.copy(states)
        # for i in range(2,len(states_og)):
        #     if states_og[i-1]-states_og[i-2] == states_og[i]-states_og[i-1]:
        #         if np.abs(states[i-1]-states[i-2]) == 1:
        #             states[i-1] = states_og[i-2] + np.sign(states_og[i-1]-states_og[i-2])*2
        #
        # Comment Flo: this function has been de-activated because of the new smoothing routine

        # Identify jumps between states and store their index
        jumps = np.concatenate((np.diff(states),np.array([0])))
        jump_idx = np.squeeze(np.argwhere(jumps!=0))

        # Create dictionary of possible tck sequences based on initial state
        possible_ticks = {}
        for i,state in enumerate(self.tick[:-1]):
            possible_ticks[state] = np.concatenate((np.roll(self.tick[:-1],shift=-i),np.array([state])))

        # Define initial state for tick as the modal state in the distribution
        # In a biased regime there will be a preferred state from which ticks originate
        state_vals,counts = np.unique(states,return_counts=True)
        state0 = np.argmax(counts)  # number k in 0,1,2
        tick = possible_ticks[state0] # tick sequence k,l,m,k
        sequence = [state0]
        tick_count, tick_time = 0, []

        # Loop through states data and create sequences of states
        for idx in jump_idx:
            # Add states to the sequence
            sequence.append(states[idx+1])
            if len(sequence) == len(tick):  # check if length is equal the tick sequence length
                # If the sequence matches the tick sequence, count a tick
                if np.sum((np.array(sequence)-tick)**2) == 0:
                    tick_count += 1
                    tick_time.append(time[idx])
                if sequence[0] == sequence[-1]:
                    # Collapse closed sequence to start / end point
                    sequence=[sequence[-1]]
                else:
                    # Collapse non-closed sequence to [start,end]
                    # not yet compatible with cycle lengths >3.
                    sequence = [sequence[0],sequence[-1]] 
                # sequence = sequence[-3:] #[states[idx+1]]
        tick_time = np.array(tick_time)

        if len(tick_time)>1:
            mean_tick_time = np.mean(np.diff(tick_time))
            std_tick_time = np.std(np.diff(tick_time))
            if self.details:
                print(f'Mean Time Between Ticks : {mean_tick_time:.3f} s')
                print(f'Standard Deviation Time Between Ticks : {std_tick_time:.3f} s')
        else:
            mean_tick_time = 0.0
            std_tick_time = 0.0
            if self.details:
                print("Standard deviation and mean could not be computed\n")

        if self.plot:
            plt.figure(figsize=(12, 5))
            plt.plot(time,states,marker=".",color="red",linestyle=None)
            plt.vlines(tick_time,ymin=-0.1*np.max(states),ymax=1.1*np.max(states),colors='b')
            plt.xlabel(r"time $[t]=\mathrm{s}$")
            plt.ylabel("state discretized")
            plt.show()

        return tick_time, tick_count, mean_tick_time, std_tick_time
    
def gauss(x,*params):
    if isinstance(params[0],np.ndarray) or isinstance(params[0],list):
        params = params[0]

    return params[-1]/np.sqrt(2*np.pi)*np.exp(-0.5 * (x-params[0])**2 / params[1]**2)
    # return 1/np.sqrt(2*np.pi)*np.exp(-0.5 * (x-params[0])**2 / params[1]**2)

def triple_point_readout(x,*params):
    # Function to be optimised to readout histogram data
    if isinstance(params[0],np.ndarray) or isinstance(params[0],list):
        params = params[0]

    y = np.zeros_like(x)
    for i in range(int(len(params)/3)):
        mean = params[i*3]
        sigma = params[i*3+1]
        amplitude = params[i*3+2]

        y = y + amplitude * np.exp(-0.5*(x-mean)**2/sigma**2)

    return y**2

def triple_point_readout_cumulative(x,*params):   
    # Function to be optimized to readout empirical
    # distribution from measurements

    if isinstance(params[0],np.ndarray) or isinstance(params[0],list):
        params = params[0]

    y = np.zeros_like(x)

    means = params[::3]
    sigmas = params[1::3]
    amplitudes = params[2::3]

    amplitudes = np.concatenate((amplitudes,np.array([1-np.sum(amplitudes)])))

    for i in range(int((len(params)+1)/3)):
        y = y + amplitudes[i] * (1+erf(1/np.sqrt(2)*(x-means[i])/sigmas[i]))/2.0

    return y

def obtain_states(folder_bias,id,mode="READ",k_smooth=3,debug=False,channel="B-V-",noisy=False,T_fridge=180e-3):
    """
    Obtain and save discretized states for given data trace.
    Input:
        folder_bias :   string ~ "0.250 bias"
        id          :   string ~ "3053"
        mode        :   string ~ "READ" or "WRITE"
        k_smooth    :   int, smoothing parameter for middle level
        debug       :   True -> plot intermediate results, False -> no plotting
        channel     :   string ~ "B-V-" for current, "PCA--" for reflectometry, others also possible
        noisy       :   False -> standard, True -> low-pass filter
        T_fridge    :   180 mK standard
    """
    # File names for the data folder
    file_names = ['dummy_parameter--',channel,'time-s-']

    # Obtain the true measurement data
    sensor_data, time, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=file_names)

    # Read DAC meta data values
    dac2 = meta_data['instrument_summary']['dac']['dac2']
    dac1 = meta_data['instrument_summary']['dac']['dac1']

    # Shift time to start at t=0
    time = time - time[0]

    # Prepare sensor data
    if channel=="B-V-":
        # CASE: current measurement
        # Rescale from A to pA
        sensor_data = sensor_data* 1e12
    elif channel=="PCA--":
        # CASE: Reflectometry
        # Rescale
        sensor_data = sensor_data* 1e3
        # Make sure current > 0
        sensor_data -= min(sensor_data)

        # Delete the 50 lowest outliers
        sensor_data_sorted = np.sort(sensor_data)
        threshold_low  = sensor_data_sorted[75]
        threshold_high = sensor_data_sorted[-1]
        idx_delete = np.concatenate((np.where(sensor_data<=threshold_low)[0],np.where(sensor_data>=threshold_high)[0]))

        for idx in idx_delete:
            sensor_data[idx] = sensor_data[idx-1]

        # time = np.delete(time,idx_delete)
        # sensor_data = np.delete(sensor_data,idx_delete)
    elif channel=="D-V-":
        # CASE: Reflectometry
        # Rescale
        sensor_data = sensor_data* 1e3

        # Delete the 50 lowest outliers
        sensor_data_sorted = np.sort(sensor_data)
        threshold_low  = sensor_data_sorted[75]
        threshold_high = sensor_data_sorted[-1]
        idx_delete = np.concatenate((np.where(sensor_data<=threshold_low)[0],np.where(sensor_data>=threshold_high)[0]))

        time = np.delete(time,idx_delete)
        sensor_data = np.delete(sensor_data,idx_delete)
    else:
        # CASE: Else
        print("note: sensor data not rescaled. consider turning on debug mode")

    # Determine entropy for sensor dot
    kBoltzmann = 1.380649e-23 # in J/K
    if channel=="B-V-":
        I0 = np.average(sensor_data)
        
        S_int = 1e-12*dac2*np.trapz(np.abs(sensor_data-I0),time) / (kBoltzmann*T_fridge)

        np.save("data/"+folder_bias+"/"+id+"/"+channel+"sensor_entropy.npy",S_int)

    # Determine entropy for reflectometry
    if channel=="PCA--":
        PWR_int = get_reflectometry_power(folder_bias,id)
        S_int = PWR_int / (kBoltzmann*T_fridge)

        np.save("data/"+folder_bias+"/"+id+"/"+channel+"sensor_entropy.npy",S_int)

    # Calibrate state readout
    if mode=="READ":
        _mode = 2
    elif mode=="WRITE":
        _mode = 1
    else:
        _mode = 0
    # mode:   0   generate opt param guess freshly
    #         1   generate and WRITE opt param guess
    #         2   READ opt parameter guess
    readout = ReadoutStates(sensor_data=sensor_data,plot=debug,ksmooth=k_smooth,nbins=180,gauss=True)
    readout.calibrate(mode=_mode,directory="data/"+folder_bias+'/'+id,noisy=noisy,channel=channel)

    # Convert readout trace to states using classifier
    states, err = readout()

    # Swap HIGH vs LOW for the PCA files
    if channel=="PCA--" and np.allclose(dac2,0.345):
        idx_low = np.where(states==0)
        idx_high = np.where(states==2)
        states[idx_low]=2
        states[idx_high]=0

    # Save Check Existence of folder
    newpath = "data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Determine SIGNAL to NOISE for readout
    popt = np.load("data/"+folder_bias+"/"+id+"/"+channel+"p0.npy")

    sensor_signal = np.zeros_like(sensor_data)  # Signal discrete, rescaled to current
    if channel == "PCA--":
        sensor_signal[np.where(states==0)] = popt[6]
        sensor_signal[np.where(states==1)] = popt[3]
        sensor_signal[np.where(states==2)] = popt[0]
    else:
        for i in range(3):            
            sensor_signal[np.where(states==i)] = popt[3*i]
    E_Xi2 = np.average((sensor_data - sensor_signal)**2)
    E_Si = np.average(sensor_signal)
    Var_Si = np.average((sensor_signal - E_Si)**2)
    np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/RO_SNR.npy",(Var_Si / E_Xi2))

    # Save data
    np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy",states)
    np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy",time)

    # Save identification error
    np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/err.npy",err)

    pass