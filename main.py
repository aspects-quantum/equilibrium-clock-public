import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy.signal import savgol_filter

from scipy.stats import expon

from data_loader import load_id

from num_stat import NumberStatistics, RateStatistics
from data_converter import ReadoutGaussian, ReadoutStates, TickCounter, obtain_states

from visualizations import *

from theory_model import DQDClock

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8
})

def data_prepro(biases,ids,k_smooth=3,debug=False,idx=[0],mode="READ",noisy=False,channel="B-V-"):
    """
    Wrapper for pre-processing data
    """
    for k, folder_bias in enumerate(biases):
        _ids = np.array(ids[k])[idx]
        for id in _ids:
            obtain_states(folder_bias,id,mode=mode,k_smooth=k_smooth,debug=debug,channel=channel,noisy=noisy)
            print("... pre-processing data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+" complete!")

def rates_postpro(biases,ids,k_smooth=3,print_res=False,idx=[0],channel="B-V-"):
    """
    Wrapper for post-processing the data into rates
    """
    for k, folder_bias in enumerate(biases):
        _ids = np.array(ids[k])[idx]
        for id in _ids:
            states = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy")
            time   = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy")

            meta_data = json.load(open("data/"+folder_bias+"/"+id+"/meta_data.json"))
            sen_bias = meta_data['instrument_summary']['dac']['dac2']
            dot_bias = meta_data['instrument_summary']['dac']['dac1']

            myRates = RateStatistics(states=states,time=time,identifier=(dot_bias,sen_bias))

            M, GammaCond, GammaMarkov, M_err = myRates.time_stats_conditional()

            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M.npy",M)
            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M_err.npy",M_err)
            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/GammaCond.npy",GammaCond)
            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/GammaMarkov.npy",GammaMarkov)

            print("... rate post-processing data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+" complete!")

            if print_res:
                print("Rate analysis result "+folder_bias+", id "+id+"\n")
                for i in range(3):
                    for j in range(3):
                        print("M"+str(j)+str(i)+" (1/sec) = "+str(np.round(M[j,i],5)))
                for i in range(3):
                    for j in range(2):
                        print("G"+str((j+i+1)%3)+str(i)+" (1/sec) = "+str(np.round(GammaCond[i,j],5))+" conditional")
                for i in range(3):
                    for j in range(2):
                        for k in range(2):
                            print("G"+str((j+i+1)%3)+str(i)+"|"+str((k+i+1)%3)+" (1/sec) = "+str(np.round(GammaMarkov[i,j,k],5))+" conditional")

def times_postpro(biases,ids,k_smooth=3,idx=[0],channel="B-V-"):
    """
    Wrapper for post-processing the data into LEFT->RIGHT and RIGHT-LEFT

    Convention:
        RIGHT -> LEFT  = 0 <- 2 <- 1 <- 0, 0210
        LEFT  -> RIGHT = 0 <- 1 <- 2 <- 0, 0120

    """
    for k, folder_bias in enumerate(biases):
        _ids = np.array(ids[k])[idx]
        for id in _ids:
            states = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy")
            time   = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy")
        
            # Create tick counter for right to left transport
            # print("\nRight -> Left")
            right_to_left = TickCounter(tick=[0, 1, 2, 0],details=False)
            right_to_left.plot = False
            rl_t, rl_n, rl_mean, rl_std = right_to_left(states, time)
            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/rl_t.npy",rl_t)

            # Create tick counter for left to right transport
            # print("\nLeft -> Right")
            left_to_right = TickCounter(tick=[0, 2, 1, 0],details=False)
            left_to_right.plot = False
            lr_t, lr_n, lr_mean, lr_std = left_to_right(states,time)
            np.save("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/lr_t.npy",lr_t)

            print("... times post-processing data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+" complete!")


def main():
    biases = [
              '0 bias',
              '+0.005 bias',
              '+0.015 bias',
              '+0.025 bias',
              '+0.205 bias',
              '+0.455 bias',
              '+0.705 bias',
              '+0.855 bias',
            #   '+0.905 bias',
              '+0.955 bias',
              ]
    # Comment: The +0.905 mV bias case has been removed from the analysis due to large drifts in the rates
    #          and because around time-tag ~1500sec the DQD got pinned in the "high" state.

    ids = [[id for id in os.listdir('data/'+folder) if os.path.isdir(os.path.join('data/'+folder, id))] for folder in biases]    

    #######################
    # DATA PRE-PROCESSING #
    #######################
    # data_prepro(biases,ids,k_smooth=3,debug=False,idx=[0,1,2],mode="READ",noisy=True,channel="B-V-")

    ########################
    # Rate matrix analysis #
    ########################
    # rates_postpro(biases,ids,k_smooth=3,idx=[0,1,2,3],print_res=False,channel="B-V-")

    ############################
    # Tunneling times analysis #
    ############################
    # times_postpro(biases,ids,k_smooth=3,idx=[0,1,2,3],channel="PCA--")

    ##################
    # Visualizations #
    ##################

    # visualize_rates(biases,ids,idx=[0],V0=0.075,k_smooth=3)
    # rate_stability_paper()
    plot_panel2(biases,ids,n_samp=300,k_smooth=3,V0_SEN=0.075,V0_DQD=0.045,channel="B-V-")
    # plot_panel1()
    # plot_endmatter()
    # plot_readout_SNR(biases,ids)
    # plot_rates_histo()

if __name__=='__main__':
    main()
