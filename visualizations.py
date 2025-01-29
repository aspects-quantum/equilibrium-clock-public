import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import json

from scipy.signal import savgol_filter

from num_stat import NumberStatistics, TimeEstimator, np_search_sequence, RateStatistics
from theory_model import DQDClock

from data_loader import load_id

from data_converter import ReadoutStates


plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8

})

col_scheme_BR = ['midnightblue','indigo','mediumvioletred','maroon','darkorange','gold','yellow','olive','darkgreen','teal',"midnightblue","mediumblue","green","seagreen","firebrick","darkred","darkorange","burlywood","purple","magenta",]

def visualize_precision(biases,ids,idx=[0],V0=0.075,n_samp=500,ev_steps=500,k_smooth=3,channel="B-V-"):
    """
    Plot the precision as a function of time
    """
    # Start empty lists
    precision_list  = []
    eval_times_list = []

    precision_theory_list   = []

    dot_bias_list   = []
    sen_bias_list   = []

    err_bar_list    = []

    # Iterate through all the biases and ids
    for k, folder_bias in enumerate(biases):
        for id in np.array(ids[k])[idx]:
            rl_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/rl_t.npy")
            lr_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/lr_t.npy")

            meta_data = json.load(open("data/"+folder_bias+"/"+id+"/meta_data.json"))
            sen_bias_list.append(meta_data['instrument_summary']['dac']['dac2'])
            dot_bias_list.append(meta_data['instrument_summary']['dac']['dac1']+V0)

            myNumStat = NumberStatistics(rl_t,lr_t,method="bi")

            t0 = 0.0
            t1 = np.max(np.concatenate((lr_t,rl_t)))/n_samp

            ev_times = np.linspace(t0,t1,ev_steps)

            prec, err = myNumStat.precision(ev_times,errorbar=True)

            precision_list.append(prec)
            eval_times_list.append(ev_times)
            err_bar_list.append(err)

            # Theoretical prediction
            M = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M.npy")

            # Start clock
            myDQDClock = DQDClock(1,[0,0],[0,0],[0,0],0)
            myDQDClock.M = M.astype(complex)

            precision_theory_list.append(myDQDClock.getAccuracy(0))

            print("... num-stat post-processing data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+" complete!")

    # Plotting
    plot_Nstat(dot_bias_list,precision_list,eval_times_list,sen_bias_list,precision_theory_list,errorbars=err_bar_list)

def plot_Nstat(bias,precision,times,dac2,precision_theory_list,errorbars=None):
    """
    Function to plot precision of N(t) for multiple biases
    """
    plt.figure(figsize=(10,3))
    if errorbars == None:
        for k, (bias_k, precision_k, times_k, dac2_k, prec_th_k) in enumerate(zip(bias,precision,times,dac2,precision_theory_list)):
            plt.plot(times_k[20:-5],precision_k[20:-5],label="bias (mV) = "+str(np.round(bias_k,3))+", dac2 = "+str(dac2_k),color=col_scheme_BR[2*k])
            plt.hlines(prec_th_k,min(times_k),max(times_k),linestyle="--",label=r"theory $\mathcal N\sim"+str(np.round(prec_th_k,2))+"$",color=col_scheme_BR[2*k])
    else:
        for k, (bias_k, precision_k, times_k, dac2_k, prec_th_k, errorbar_k) in enumerate(zip(bias,precision,times,dac2,precision_theory_list,errorbars)):
            plt.plot(times_k[20:-5],precision_k[20:-5],label="bias (mV) = "+str(np.round(bias_k,3))+", dac2 = "+str(dac2_k))
            plt.plot(times_k[20:-5],savgol_filter((precision_k+(errorbar_k))[20:-5],50,3),color="black",linewidth=0.8,linestyle="--",label="error-tube")
            plt.plot(times_k[20:-5],savgol_filter((precision_k-(errorbar_k))[20:-5],50,3),color="black",linewidth=0.8,linestyle="--")
            plt.hlines(prec_th_k,min(times_k),max(times_k),linestyle="--",label=r"theory $\mathcal N\sim"+str(np.round(prec_th_k,2))+"$",color=col_scheme_BR[2*k])

    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.xlabel(r"time $t=[s]$")
    plt.ylabel(r"precision $\langle N(t) \rangle / \mathrm{Var}[N(t)]$")
    plt.savefig("figs/precision.pdf")#,dpi=400)
    plt.show()

    pass

def plotMrates(bias,M,M_err,channels):
    fig, axs = plt.subplots(1,3,figsize=(7.08333,2.0),sharex=True,sharey=True)
    axs = np.atleast_2d(axs)

    # Labels
    lbls = [" (dc)"," (rf)"]
    markers = [".","*"]
    linestyles = ["-",":"]
    alphas = [1,1]

    # Note: This is for plotting i<-j vs j<-i
    for j, channel in enumerate(channels):
        # Unpack matrices
        _M = np.stack(M[j],axis=0)
        _M_err = np.stack(M_err[j],axis=0)

        # Plotting
        for i in range(3):
            axs[0,i].errorbar(bias,_M[:,i,(i+1)%3],yerr=_M_err[:,i,(i+1)%3],capsize=3.0,
                              marker=markers[j],linestyle=linestyles[j],color=col_scheme_BR[0],
                              alpha=alphas[j],label=r"$M_{"+str(i+1)+str((i+1)%3+1)+r"}$"+lbls[j])
            axs[0,i].errorbar(bias,_M[:,(i+1)%3,i],yerr=_M_err[:,(i+1)%3,i],capsize=3.0,
                              marker=markers[j],linestyle=linestyles[j],color=col_scheme_BR[2],
                              alpha=alphas[j],label=r"$M_{"+str((i+1)%3+1)+str(i+1)+r"}$"+lbls[j])
            axs[0,i].legend(loc="upper left")
            if i==0:
                axs[0,i].set_ylabel(r"rate (Hz)")
            axs[0,i].set_xlabel(r"$V_{\rm DQD}$ (mV)")
        # axs[0,i].grid(color="black",alpha=0.1,linestyle="--")

    plt.tight_layout()
    plt.savefig("figs/paper/rates_vs_DQD_voltage.pdf")#,dpi=600)
    # plt.close()
    plt.show()
    pass

def visualize_rates(biases,ids,idx=[0],V0=0.075,k_smooth=3):
    """
    Plots the rates in the rate matrix
    """
    channels = ["B-V-","PCA--"]

    for k in idx:
        M_dc_list     = []
        M_rf_list     = []
        M_lists       = [M_dc_list,M_rf_list]
        M_dc_err_list = []
        M_rf_err_list = []
        M_err_lists   = [M_dc_err_list,M_rf_err_list]
        bias_list     = []

        for m,folder_bias in enumerate(biases):
            id = ids[m][k]

            # Cycle through two methods
            for j,channel in enumerate(channels):
                # Read out rate matrix
                _M = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M.npy")
                _M_err = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M_err.npy")
                M_lists[j].append(_M)
                M_err_lists[j].append(_M_err)

            # Read out bias
            meta_data = json.load(open("data/"+folder_bias+"/"+id+"/meta_data.json"))
            bias_k = meta_data['instrument_summary']['dac']['dac1']+V0
            bias_list.append(bias_k)

        plotMrates(bias_list,M_lists,M_err_lists,channels)

def plot_panel2(biases,ids,n_samp,k_smooth=3,T_fridge=150e-3,V0_SEN=0.075,V0_DQD=-0.045,idx=[0,1,2,3],channel="B-V-"):
    """
    Plot steady-state precision as a function of entropy production

        T_fridge    :   temperature of fridge in units Kelvin (K)
    """
    # Natural constants
    kB = 8.617333262e-5 # eV / K

    # Empty arrays for precision
    # Indices: [net/opt,DQD Bias,Sen Bias]
    precision_stat      = np.zeros((2,len(biases),len(idx)))    # Experimental data DC
    precision_stat_rf   = np.zeros((2,len(biases),len(idx)))    # Experimental data RF
    precision_model     = np.zeros((2,len(biases),len(idx)))    # Model DC
    precision_model_rf  = np.zeros((2,len(biases),len(idx)))    # Model RF

    # Errors
    error_stat      = np.zeros((2,len(biases),len(idx)))
    error_stat_rf   = np.zeros((2,len(biases),len(idx)))
    error_model     = np.zeros((2,len(biases),len(idx)))
    error_model_rf  = np.zeros((2,len(biases),len(idx)))

    # Entropy in DQD
    entropy = np.zeros((len(biases),len(idx)))

    dac2s = np.zeros((len(idx)))

    n_dot_bias = len(biases)    # Number of dot bias settings
    n_sen_bias = len(idx)       # Number of sensor bias settings

    # Entropy in Sensor CURRENT
    entropy_list = np.zeros((n_dot_bias,n_sen_bias))

    # Entropy in Sensor CURRENT
    reflecto_entropy = np.zeros((n_dot_bias,n_sen_bias))

    dot_bias = np.zeros((len(biases)))

    # Loop through all measurements
    for kDot, folder_bias in enumerate(biases):
        meta_data = json.load(open("data/"+folder_bias+"/"+ids[kDot][idx[0]]+"/meta_data.json"))
        dot_bias[kDot] = meta_data['instrument_summary']['dac']['dac1']+V0_SEN

        for kSen, id in enumerate(np.array(ids[kDot])[idx]):
            # Load experimental data
            states      = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy")
            states_rf   = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/states.npy")
            times  = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy")
            times_rf  = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/time.npy")

            rl_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/rl_t.npy")
            lr_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/lr_t.npy")
            rl_t_rf = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/rl_t.npy")
            lr_t_rf = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/lr_t.npy")
            S_tot = np.load("data/"+folder_bias+"/"+id+"/"+channel+"sensor_entropy.npy")
            S_reflecto = np.load("data/"+folder_bias+"/"+id+"/"+"PCA--"+"sensor_entropy.npy")

            # Determine entropy of Sensor dot
            entropy_list[kDot,kSen] = S_tot/(times[-1]-times[0]) #np.abs(len(rl_t)-len(lr_t))

            # Load entropy rate for reflectometry
            reflecto_entropy[kDot,kSen] = S_reflecto

            # Determine entropy of DQD
            meta_data = json.load(open("data/"+folder_bias+"/"+id+"/meta_data.json"))
            deltaV = meta_data['instrument_summary']['dac']['dac1']

            DAC2 = meta_data['instrument_summary']['dac']['dac2']
            dac2s[kSen] = DAC2

            entropy[kDot,kSen] = np.abs(deltaV+V0_DQD)*1e-3 / (kB * T_fridge)
            print("V (V) :",np.abs(deltaV+V0_DQD)*1e-3)
            print("T (K) :",T_fridge)

            ################################
            # Precision of the NET counter #
            ################################
            try:
                myNumStat = NumberStatistics(rl_t,lr_t,method="bi")
                myNumStat_rf = NumberStatistics(rl_t_rf,lr_t_rf,method="bi")

                t0 = 0.0
                t1 = np.max(np.concatenate((lr_t,rl_t)))/n_samp

                ev_times = np.linspace(t0,t1,20)

                prec, err = myNumStat.precision(ev_times,errorbar=True,estimator=True)
                prec_rf, err_rf = myNumStat_rf.precision(ev_times,errorbar=True,estimator=True)

                precision_stat[0,kDot,kSen]     = prec[-2]
                precision_stat_rf[0,kDot,kSen]  = prec_rf[-2]
                error_stat[0,kDot,kSen]         = err[-2]
                error_stat_rf[0,kDot,kSen]      = err_rf[-2]
            except Exception as error:
                print("failor to compute net estimator for \n"
                      +folder_bias+" | "+id+" | "+channel+"\n")
                print(error)

            try:
                # Theoretical prediction
                M_dc = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/M.npy")
                M_err_dc = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/M_err.npy")
                M_rf = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/M.npy")
                M_err_rf = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(k_smooth)+"/M_err.npy")

                # Start clock
                myDQDClock = DQDClock(1,[0,0],[0,0],[0,0],0)
                myDQDClock.M = M_dc.astype(complex)

                precision_model[0,kDot,kSen] = myDQDClock.getAccuracy(0) * (len(lr_t)-len(rl_t))/np.max(np.concatenate((lr_t,rl_t)))

                myDQDClock_rf = DQDClock(1,[0,0],[0,0],[0,0],0)
                myDQDClock_rf.M = M_rf.astype(complex)

                precision_model_rf[0,kDot,kSen] = myDQDClock_rf.getAccuracy(0) * (len(lr_t)-len(rl_t))/np.max(np.concatenate((lr_t,rl_t)))

                # Error for Net estimator
                myErrorClock = DQDClock(1,[0,0],[0,0],[0,0],0)
                def _SNR(rates):
                    """
                    M = [[X, r0, r1]
                         [r2, Y, r3]
                         [r4, r5, Z]]
                    """
                    _M_tmp = np.array([[-rates[2]-rates[4], rates[0], rates[1]],
                                       [rates[2], -rates[0]-rates[5], rates[3]],
                                       [rates[4], rates[5], -rates[1]-rates[3]]],dtype=complex)
                    myErrorClock.M = _M_tmp
                    return myErrorClock.getAccuracy(0) * (len(lr_t)-len(rl_t))/np.max(np.concatenate((lr_t,rl_t)))
                
                def gaussError(fun,x,dx):
                    """
                    Return gauss error propagated error
                    """
                    x  = np.atleast_1d(x)
                    dx = np.atleast_1d(dx)
                    res = 0.0
                    for i, dxi in enumerate(dx):
                        ei = np.zeros((len(x)))
                        ei[i] = 1.0
                        Dfi = (fun(x+ei*1e-5) - fun(x-ei*1e-5))/2e-5
                        res += Dfi**2 * dxi**2
                    return np.sqrt(res)
                
                def M_reducer(_M):
                    return np.array([_M[0,1],_M[0,2],_M[1,0],_M[1,2],_M[2,0],_M[2,1]])

                error_model[0,kDot,kSen] = gaussError(_SNR,M_reducer(M_dc),M_reducer(M_err_dc))

                #################################
                # Precision of the BLUE counter #
                #################################
                if kSen<3:
                    myEstimator = TimeEstimator(times,states,M_dc)
                    myEstimator_rf = TimeEstimator(times_rf,states_rf,M_rf)

                    # Determine theoretical precision
                    v0 = myDQDClock.getSteadyState()
                    meanRT = np.sum(np.abs(v0/np.diag(M_dc)))
                    precision_model[1,kDot,kSen] = np.real(1/meanRT)
                    precision_stat[1,kDot,kSen], error_stat[1,kDot,kSen] = myEstimator.getLateSNR(
                        n_samp,errorbar=True)
                    
                    # Error bar model
                    def _SNR_TE(rates):
                        """
                        M = [[X, r0, r1]
                            [r2, Y, r3]
                            [r4, r5, Z]]
                        """
                        _M_tmp = np.array([[-rates[2]-rates[4], rates[0], rates[1]],
                                        [rates[2], -rates[0]-rates[5], rates[3]],
                                        [rates[4], rates[5], -rates[1]-rates[3]]],dtype=complex)
                        myErrorClock.M = _M_tmp
                        _v0 = myErrorClock.getSteadyState()
                        _meanRT = np.sum(np.abs(_v0/np.diag(_M_tmp)))
                        return np.real(1/_meanRT)

                    error_model[1,kDot,kSen] = gaussError(_SNR_TE,M_reducer(M_dc),M_reducer(M_err_dc))
                    # error_model[1,kDot,kSen] = np.sqrt(np.sum(np.abs(np.diag(M_err_dc))**2 * v0**2 / np.diag(M_dc)**4)) / meanRT**2
                    
                    v0_rf = myDQDClock_rf.getSteadyState()
                    meanRT_rf = np.sum(np.abs(v0_rf/np.diag(M_rf)))
                    precision_model_rf[1,kDot,kSen] = np.real(1/meanRT_rf)
                    precision_stat_rf[1,kDot,kSen], error_stat_rf[1,kDot,kSen] = myEstimator_rf.getLateSNR(
                        n_samp,errorbar=True)
            except Exception as error:
                print("failure to compute theoretical precision for \n"
                      +folder_bias+" | "+id+" | "+channel+"\n")
                print(error)

            print("... num-stat post-processing data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+" complete!")

    fig, axs = plt.subplots(1,3,figsize=(7.08333,2.0))


    # SUBFIG A)
    idx_sort = np.argsort(entropy[:,0])

    axs0R = axs[0].twinx()

    axs[0].errorbar(entropy[idx_sort,1],precision_stat[1,idx_sort,0],yerr=error_stat[1,idx_sort,0],
                    capsize=3,color=col_scheme_BR[0],marker=".",label=r"$\Theta_{\rm opt}$")
    axs[0].errorbar(entropy[idx_sort,1],precision_model[1,idx_sort,0],yerr=error_model[1,idx_sort,0],
                    capsize=3,color=col_scheme_BR[1],linestyle="",alpha=0.7,marker="*",label=r"$\Theta_{\rm opt}^{\rm th.}$")
    axs0R.errorbar(entropy[idx_sort,0],precision_stat[0,idx_sort,0],yerr=error_stat[0,idx_sort,0],
                    capsize=3,color=col_scheme_BR[2],marker=".",label=r"$\Theta_{\rm net}$")
    axs0R.errorbar(entropy[idx_sort,0],precision_model[0,idx_sort,0],yerr=error_model[0,idx_sort,0],
                    capsize=3,color=col_scheme_BR[3],linestyle="",alpha=0.7,marker="*",label=r"$\Theta_{\rm net}^{\rm th.}$")
    # axs[0].set_yscale("log")
    axs[0].set_xlabel(r"entropy per tick $\Sigma_{\rm tick}/k_B$")
    axs[0].set_ylabel(r"precision $\mathcal S$ (Hz)")

    axs[0].text(-0.25, 0.94, "(a)",transform=axs[0].transAxes)
    axs[0].set_ylim(-5,32)
    axs0R.set_ylim(-0.5,3.2)
    # axs[0].set_ylim(-4,14)
    # axs0R.set_ylim(-0.004,0.014)
    axs[0].legend(loc="upper left")
    axs0R.legend(loc="lower right")

    # SUBFIG B)
    axs[1].errorbar(entropy_list[-2,:-1],precision_stat[1,-2,:-1],yerr=error_stat[1,-2,:-1],
                    capsize=3,color=col_scheme_BR[0],marker=".",label=r"$\Theta_{\rm opt}$")
    axs[1].errorbar(entropy_list[-2,:],precision_stat[0,-2,:],yerr=error_stat[0,-2,:],
                    capsize=3,color=col_scheme_BR[2],marker=".",label=r"$\Theta_{\rm net}$")

    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"entropy current $\dot \Sigma$ ($k_B$/s)")
    # axs[1].set_ylim(8e-4,8e1)

    axs[1].text(-0.15, 0.94, "(b)",transform=axs[1].transAxes)

    idx_sort = np.argsort(reflecto_entropy[-3,:])

    # SUBFIG C)
    axs[2].errorbar(reflecto_entropy[-2,idx_sort[1:]],precision_stat_rf[1,-2,idx_sort[1:]],
                    yerr=error_stat_rf[1,-2,idx_sort[1:]],capsize=3.0,
                    color=col_scheme_BR[0],marker=".",label=r"$\Theta_{\rm opt}$")
    axs[2].errorbar(reflecto_entropy[-2,idx_sort],precision_stat_rf[0,-2,idx_sort],
                    yerr=error_stat_rf[0,-2,idx_sort],capsize=3.0,
                    color=col_scheme_BR[2],marker=".",label=r"$\Theta_{\rm net}$")

    axs[2].set_yscale("log")
    axs[2].set_xscale("log")
    axs[2].set_xlabel(r"entropy reflect. $\dot \Sigma$ ($k_B$/s)")
    # axs[2].set_xticks([3e10,4e10,5e10],minor=True)
    # axs[2].minorticks_off()

    axs[2].text(-0.15, 0.94, "(c)",transform=axs[2].transAxes)


    for ax in axs:
        ax.legend(loc="lower right")
        # ax.grid(linewidth="0.6",linestyle="--",alpha=0.5,color="gray")
    
    axs[0].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("figs/paper/panel2.svg")
    plt.show()
    # plt.close()

    pass

def plot_panel1():


    folder_bias = "+0.855 bias"
    id = "5548"

    sensor_data, time, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"B-V-",'time-s-'])
    sensor_data_rf, _t, _m = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"D-V-",'time-s-'])

    # sensor_data_rf = -(sensor_data_rf - np.average(sensor_data_rf)) * np.std(sensor_data) / np.std(sensor_data_rf) + np.average(sensor_data)

    states = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/states.npy")
    times  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/time.npy")
    rl_t = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/rl_t.npy")
    lr_t  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/lr_t.npy")

    dT = times[1]-times[0]

    idxTest = np.where(time==rl_t[10])[0][0]

    idxStart = idxTest-87
    idxEnd  = idxTest+55

    print(idxStart)
    print(idxEnd)

    p0     = np.load("data/"+folder_bias+"/"+id+"/B-V-p0.npy")

    # Discretized states renormalize
    idx0 = np.where(states==0)
    idx1 = np.where(states==1)
    idx2 = np.where(states==2)
    states[idx0] = p0[0]
    states[idx1] = p0[3]
    states[idx2] = p0[6]

    mpl.rcParams['axes.linewidth'] = 0.75
    fig, axs = plt.subplots(1,1,figsize=(4,1.1),sharex=True,gridspec_kw={'hspace': 0})

    # (ax1,ax2) = axs
    ax1 = axs
    ax2 = ax1.twinx()

    ymax = max(1e12*sensor_data[idxStart:idxEnd])*1.01
    ymin = min(1e12*sensor_data[idxStart:idxEnd])/1.01

    ax1.vlines(lr_t-time[idxStart-1]-dT/2,linestyle="--",linewidth=0.5,ymin=ymin,ymax=ymax,color="red")
    # ax1.plot(lr_t-time[idxStart-1],np.ones(len(lr_t))*(ymax-2),linestyle="",marker="+",markerwidth=5,color="red")
    ax1.vlines(rl_t-time[idxStart-1]-dT/2,linestyle="--",linewidth=0.5,ymin=ymin,ymax=ymax,color="blue")
    # ax1.plot(rl_t-time[idxStart-1],np.ones(len(rl_t))*(ymax-2),linestyle="",marker="*",markersize=5,color="blue")

    ax1.plot(time[idxStart:idxEnd]-time[idxStart],1e12*sensor_data[idxStart:idxEnd],
             linestyle="-",linewidth=0.75,color=col_scheme_BR[0],label=r"$I_{\rm dc}$")
    ax2.plot(time[idxStart:idxEnd]-time[idxStart],1e3*sensor_data_rf[idxStart:idxEnd],
             linestyle="-",linewidth=0.75,color=col_scheme_BR[2],label=r"$\propto V_{\rm rf}$")
    ax1.stairs(states[idxStart:idxEnd],time[idxStart-1:idxEnd]-time[idxStart]+dT/2,linewidth=0.75,color="black")#,label=r"$s(t)$")

    ax1.set_ylim(ymin,ymax)
    ax1.set_xlim(0,time[idxEnd]-time[idxStart])
    # ax1.grid(linewidth="0.6",linestyle="--",alpha=0.5,color="gray")

    # idx = np.where(((lr_t-time[idxStart-1])>0)*((lr_t-time[idxStart-1])<time[idxEnd]-time[idxStart]))
    # tick_plus = ((lr_t-time[idxStart-1])[idx]).tolist()
    # lbl_plus = [r"$+$" for k in tick_plus]

    # idx = np.where(((rl_t-time[idxStart-1])>0)*((rl_t-time[idxStart-1])<time[idxEnd]-time[idxStart]))
    # tick_minus = ((rl_t-time[idxStart-1])[idx]).tolist()
    # lbl_minus = [r"$-$" for k in tick_minus]

    # tick_already = list(ax1.get_xticks())
    # lbl_already = list((ax1.get_xticks()))

    # all_ticks = tick_plus+tick_minus+tick_already
    # all_labels = lbl_plus+lbl_minus+lbl_already

    # # idx = np.argsort(all_ticks)

    # # ordered_labels = (lbl_plus+lbl_minus+lbl_already)
    

    # ax1.set_xticks(all_ticks,all_labels)

    # ax1.set_xlim(0,time[idxEnd]-time[idxStart])


    ax1.set_ylabel(r"$I$ (pA)")
    ax2.set_ylabel(r"$V$ (mV)")
    # ax1.legend(loc="lower right")

    # Plot stairs
    idxLR = np.min(np.where(lr_t > time[idxStart]))

    N_of_t = np.arange(16)
    N_of_t[6:] -= 2

    jumpytime = np.concatenate(([0],lr_t[idxLR:idxLR+15]-time[idxStart-1],[rl_t[5]-time[idxStart-1]]))
    jumpytime = np.sort(jumpytime)

    # ax2.vlines(lr_t-time[idxStart-1],ymin=-1,ymax=8,color="black",label=r"$+1$")
    # ax2.vlines(rl_t-time[idxStart-1],ymin=-1,ymax=8,color="gray",linestyle="--",label=r"$-1$")
    # ax2.stairs(N_of_t,jumpytime,color=col_scheme_BR[2],linewidth=1.5)

    # ax2.legend(loc="lower right")

    # ax2.set_yticks([0,2,4,6])

    # ax2.set_ylabel(r"$N(t)$")
    # ax2.set_xlabel(r"reference time $t$ (s)")
    # ax2.grid(axis="y",linewidth="0.6",linestyle="--",alpha=0.5,color="gray")

    # ax2.set_ylim(-1,8)

    ax1.set_xlabel(r"reference time $t$ (s)")


    plt.tight_layout()
    plt.savefig("figs/paper/panel1.pdf")#,dpi=600)
    plt.show()

    pass

def plot_readout_SNR(biases,ids,k_smooth=3,V0_DQD=0.075,V0_SEN=0.0):#-0.045):
    """
    Instructive visualization for LMH identification
        (a) HISTO       (b) Current         (c) SNR

    """
    folder_bias = "+0.455 bias"
    id = "5537"

    sensor_data, time, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"B-V-",'time-s-'])

    states = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/states.npy")
    times  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/time.npy")
    rl_t = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/rl_t.npy")
    lr_t  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/lr_t.npy")

    idxTest = 3600 # np.where(time==rl_t[5])[0][0]

    idxStart = idxTest-150
    idxEnd  = idxTest+150

    dT = times[1]-times[0]

    print(idxStart)
    print(idxEnd)

    p0     = np.load("data/"+folder_bias+"/"+id+"/B-V-p0.npy")

    fig, axs = plt.subplots(1,3,figsize=(7.08333,2.0),width_ratios=[1,3,3])#,gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

    ymax = max(1e12*sensor_data[idxStart:idxEnd])*1.01
    ymin = min(1e12*sensor_data[idxStart:idxEnd])/1.01

    #############
    # HISTOGRAM #
    #############
    axs[0].hist(sensor_data*1e12,bins=35,density=True,color=col_scheme_BR[0],orientation='horizontal')
    axs[0].set_ylim(-0.1,3.1)
    # axs[0].set_xlim(0,0.19)
    axs[0].set_ylabel(r"$I$ (pA)")
    axs[0].set_xlabel(r"prob.\ (arb.\ units)")

    # axs[0].hlines([p0[0],p0[3],p0[6]],xmin=0,xmax=0.19,linewidth=1,linestyle="--",color="gray")

    axs[0].text(-0.4, 0.94, "(a)",transform=axs[0].transAxes)

    #############
    # CURRENT   #
    #############
    axs[1].plot(time[idxStart:idxEnd]-time[idxStart],1e12*sensor_data[idxStart:idxEnd],
             linewidth=1.0,color=col_scheme_BR[2],label=r"$I(t)$")
    # axs[1].set_xticks(ticks=[])
    axs[1].set_ylim(-0.1,3.1)
    axs[1].set_xlim(0,time[idxEnd]-time[idxStart]-dT)
    axs[1].set_xlabel(r"time $t$ (s)")
    # axs[0].sharey(axs[1])
    # axs[1].set_yticks(ticks=[0,1,2,3],labels=[])
    axs[1].set_yticklabels([])

    # axs[1].hlines([p0[0],p0[3],p0[6]],xmin=0,xmax=time[idxEnd]-time[idxStart]-dT,linewidth=1,linestyle="--",color="gray")

    axs[1].text(-0.1, 0.94, "(b)",transform=axs[1].transAxes)


    #############
    # SNR       #
    #############
    dot_bias = np.zeros((len(biases)))
    sen_bias = np.zeros((4))

    SNRs = np.zeros((len(biases),4,2))

    # Entropy in Sensor CURRENT
    entropy_list = np.zeros((len(biases),4))

    # Entropy in Sensor CURRENT
    reflecto_entropy = np.zeros((len(biases),4))


    channels = ["B-V-","PCA--"]

    for kDot, folder_bias in enumerate(biases):
        meta_data = json.load(open("data/"+folder_bias+"/"+ids[kDot][0]+"/meta_data.json"))
        dot_bias[kDot] = meta_data['instrument_summary']['dac']['dac1']+V0_DQD

        for kSen, id in enumerate(np.array(ids[kDot])):
            if kDot == 0:
                # Sensor bias
                meta_data = json.load(open("data/"+folder_bias+"/"+ids[kDot][kSen]+"/meta_data.json"))
                sen_bias[kSen] = meta_data['instrument_summary']['dac']['dac2']+V0_SEN

            # Load experimental data
            for k_ch, channel in enumerate(channels):
                RO_SNR  = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/RO_SNR.npy")

                SNRs[kDot,kSen,k_ch] = np.average(RO_SNR)

            rl_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/rl_t.npy")
            lr_t = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/lr_t.npy")
            S_tot = np.load("data/"+folder_bias+"/"+id+"/"+channel+"sensor_entropy.npy")
            S_reflecto = np.load("data/"+folder_bias+"/"+id+"/"+"PCA--"+"sensor_entropy.npy")

            # Determine entropy of Sensor dot
            # entropy_list[kDot,kSen] = S_tot/max(np.concatenate((rl_t,lr_t))) #np.abs(len(rl_t)-len(lr_t))
            # Load entropy rate for reflectometry
            reflecto_entropy[kDot,kSen] = S_reflecto

    # dc axis
    axs[2].errorbar(sen_bias,np.average(SNRs[:,:,0],axis=0),yerr=np.std(SNRs[:,:,0],axis=0,ddof=1),
                    capsize=3,color=col_scheme_BR[0],marker=".",label="dc")
    axs[2].errorbar([],[],yerr=[],
                    capsize=3,color=col_scheme_BR[2],marker="*",label="rf")
    axs[2].legend(loc="upper left")
    axs[2].set_xlabel(r"bias $V_{\rm DQD}$ (mV)")
    axs[2].set_yticks([0,5,10,15])

    # rf axis
    ax_rf = axs[2].twiny()

    rf_freq = np.array([128e6,116e6,115e6,114e6])[::-1]
    ax_rf.errorbar(rf_freq*1e-6,np.average(SNRs[:,:,1],axis=0),yerr=np.std(SNRs[:,:,1],axis=0,ddof=1),
                    capsize=3,color=col_scheme_BR[2],marker="*",label="rf")
    ax_rf.set_xlabel(r"frequency $f_{\rm rf}$ (MHz)")
    ax_rf.invert_xaxis()
    ax_rf.set_xticks(np.arange(114,129,2)[::-1])

    # axs[2].stairs(states[idxStart:idxEnd],time[idxStart:idxEnd+1]-time[idxStart]-dT/2,color=col_scheme_BR[4],label=r"$s(t)$")
    # axs[2].set_ylim(-0.5,2.5)
    # axs[2].set_xticks(ticks=[1,2,3,4]) # ,labels=["0","L","R"])
    # axs[2].set_xlim(0,time[idxEnd]-time[idxStart]-dT)
    axs[2].set_ylabel(r"SNR")

    axs[2].text(-0.1, 0.94, "(c)",transform=axs[2].transAxes)

    plt.tight_layout()
    plt.savefig("figs/paper/readout_SNR.svg")#,dpi=600)
    plt.show()

    pass

def plot_rates_histo(k_smooth=3):
    """
    Instructive visualization for rates
    (a) leaving 0   (b) leaving 1   (c) leaving 2
    """
    # INITIAL PARAMETERS
    folder_bias = "+0.455 bias" # "+0.955 bias" # "+0.905 bias" # 
    id = "5534" # "5316" # "5278" # 

    n_bins = 11

    markers = [".","x","*","<"]
    indices = ["(a)","(b)","(c)"]

    sensor_data, time, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"B-V-",'time-s-'])

    states = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/states.npy")
    times  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/time.npy")
    M      = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/M.npy")
    M_err  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(k_smooth)+"/M_err.npy")


    transitions = np.array([[1,0,1],
                            [2,0,1],
                            [1,0,2],
                            [2,0,2],
                            [0,1,0],
                            [2,1,0],
                            [0,1,2],
                            [2,1,2],
                            [0,2,0],
                            [1,2,0],
                            [0,2,1],
                            [1,2,1]])

    myRates = RateStatistics(states,time,0)
    stats = myRates.get_time_stats(transitions)

    transitions = transitions.reshape((3,4,3))

    idxr = lambda a,b : int(4*a + b)

    fig, axs = plt.subplots(1,3,figsize=(7.08333,2.0),sharey=True)#,gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

    # Plot all three transitions
    for k in range(3):
        x_hists = np.zeros((4,n_bins+1))
        y_hists = np.zeros((4,n_bins))
        
        # Do histograms
        for j in range(4):
            print(len(stats[idxr(k,j)]))
            y_hists[j,:], x_hists[j,:] = np.histogram(stats[idxr(k,j)],n_bins,density=True)
        
        # Obtain deadtime and rate
        T_dead = min(np.concatenate((stats[idxr(k,0)],stats[idxr(k,1)],stats[idxr(k,2)])))
        Gamma = -M[k,k]

        t_range = np.linspace(0,max(np.ravel(x_hists)),100)

        # Plotting
        # axs[k].set_title(r"$\widehat \Gamma_"+str(k)+"="+str(np.round(Gamma,1))+"("+str(np.round(M_err[k,k],1))+")$ Hz")
        axs[k].semilogy(t_range,Gamma * np.exp(-Gamma*(t_range - T_dead)),color=col_scheme_BR[0])
        for j in range(4):
            axs[k].semilogy(x_hists[j,:-1],y_hists[j,:],marker=markers[j],markersize=4,linestyle="",color=col_scheme_BR[1+j],
                            label=r"$"+str(transitions[k,j,0]+1)+"\leftarrow"+str(transitions[k,j,1]+1)+"|"+str(transitions[k,j,2]+1)+"$")

        axs[k].legend(loc="upper right")
        axs[k].set_xlabel(r"time $t$ (s)")
        if k==0:
            axs[k].text(-0.27, 0.93, indices[k],transform=axs[k].transAxes)
        else:
            axs[k].text(-0.12, 0.93, indices[k],transform=axs[k].transAxes)

        axs[k].text(0.05, 0.05, r"$\widehat \Gamma_"+str(1+k)+"="+str(np.round(Gamma,1))+"\pm"+str(np.round(M_err[k,k],1))+"$ Hz",transform=axs[k].transAxes)

    axs[0].set_ylabel(r"PDF (Hz)")

    plt.tight_layout()
    plt.savefig("figs/paper/rates_histo.svg")#,dpi=600)
    plt.show()

    pass

def plot_endmatter():
    """
    Instructive visualization for LMH identification
    (a) HISTO       (b) Current
    (c) HISTO       (d) rf
                    (c) States    
    """

    # folder_bias = "+0.705 bias"
    # id = "5514"

    folder_bias = "0 bias"
    id = "5197"

    sensor_data, time, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"B-V-",'time-s-'])
    sensor_data_rf, _t, _m = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"PCA--",'time-s-'])
    sensor_data_rfX, _t, _m = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"C-V-",'time-s-'])
    sensor_data_rfY, _t, _m = load_id("data/"+folder_bias,id,data_format=None,file_names=['dummy_parameter--',"D-V-",'time-s-'])


    states = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/states.npy")
    states_rfY = np.load("data/"+folder_bias+"/"+id+"/PCA--K"+str(3)+"/states.npy")
    times  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/time.npy")
    rl_t = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/rl_t.npy")
    lr_t  = np.load("data/"+folder_bias+"/"+id+"/B-V-K"+str(3)+"/lr_t.npy")

    idxTest = 1000 #np.where(time==rl_t[5])[0][0]

    idxStart = idxTest+100
    idxEnd  = idxTest+600

    dT = times[1]-times[0]

    print(idxStart)
    print(idxEnd)

    p0     = np.load("data/"+folder_bias+"/"+id+"/B-V-p0.npy")
    p0_rfY     = np.load("data/"+folder_bias+"/"+id+"/D-V-p0.npy")

    fig, axs = plt.subplots(3,2,figsize=(7.08333,3.5),width_ratios=[1,4],gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

    ymax = max(1e12*sensor_data[idxStart:idxEnd])*1.01
    ymin = min(1e12*sensor_data[idxStart:idxEnd])/1.01

    ################
    # HISTOGRAM DC #
    ################
    axs[0,0].hist(sensor_data*1e12,bins=35,density=True,color=col_scheme_BR[0],orientation='horizontal')
    axs[0,0].set_ylim(168,192)
    axs[0,0].set_xlim(0,0.27)
    axs[0,0].set_ylabel(r"$I_{\rm cs}$ (pA)")
    axs[0,0].set_xlabel(r"prob.")

    axs[0,0].hlines([p0[0],p0[3],p0[6]],xmin=0,xmax=0.4,linewidth=1,linestyle="--",color="gray")

    axs[0,0].text(-0.35, 0.88, "(a)",transform=axs[0,0].transAxes)
    axs[0,0].set_xticks([])

    ################
    # HISTOGRAM rf #
    ################
    rf_min = -3
    rf_max = 23
    axs[1,0].hist(sensor_data_rfY[np.argwhere((sensor_data_rfY*1e3<rf_max)*(sensor_data_rfY*1e3>rf_min))]*1e3,
                  bins=35,density=True,color=col_scheme_BR[2],orientation='horizontal')
    axs[1,0].set_ylim(rf_min,rf_max)
    axs[1,0].set_xlim(0,0.27)
    axs[1,0].set_ylabel(r"$V_{\rm rf}^{\rm Y}$ (mV)")
    axs[1,0].set_xlabel(r"prob.")

    axs[1,0].hlines([p0_rfY[0],p0_rfY[3],p0_rfY[6]],xmin=0,xmax=0.4,linewidth=1,linestyle="--",color="gray")

    axs[1,0].text(-0.35, 0.88, "(c)",transform=axs[1,0].transAxes)


    ############
    # DC TRACE #
    ############
    axs[0,1].plot(time[idxStart:idxEnd]-time[idxStart],1e12*sensor_data[idxStart:idxEnd],
             linewidth=1.0,color=col_scheme_BR[0],label=r"$I(t)$")
    axs[0,1].set_yticks(ticks=[])
    axs[0,1].set_xticks(ticks=[])
    axs[0,1].set_ylim(168,192)
    axs[0,1].set_xlim(0,time[idxEnd]-time[idxStart]-dT)

    axs[0,1].hlines([p0[0],p0[3],p0[6]],xmin=0,xmax=time[idxEnd]-time[idxStart]-dT,linewidth=1,linestyle="--",color="gray")

    axs[0,1].text(0.03, 0.88, "(b)",transform=axs[0,1].transAxes)

    ############
    # rf TRACE #
    ############
    axs[1,1].plot(time[idxStart:idxEnd]-time[idxStart],1e3*sensor_data_rfY[idxStart:idxEnd],
             linewidth=1.0,color=col_scheme_BR[2],label=r"$V_{\rm rf}^{\rm Y}(t)$")
    axs[1,1].set_yticks(ticks=[])
    axs[1,1].set_xticks(ticks=[])
    axs[1,1].set_ylim(rf_min,rf_max)
    axs[1,1].set_xlim(0,time[idxEnd]-time[idxStart]-dT)

    axs[1,1].hlines([p0_rfY[0],p0_rfY[3],p0_rfY[6]],xmin=0,xmax=time[idxEnd]-time[idxStart]-dT,linewidth=1,linestyle="--",color="gray")

    axs[1,1].text(0.03, 0.88, "(d)",transform=axs[1,1].transAxes)

    #############
    # LOW RIGHT #
    #############
    axs[2,1].stairs(states_rfY[idxStart:idxEnd],time[idxStart:idxEnd+1]-time[idxStart]-dT/2,
                color=col_scheme_BR[2],label=r"$S_{\rm rf}(t)$",alpha=0.5)
    axs[2,1].stairs(states[idxStart:idxEnd],time[idxStart:idxEnd+1]-time[idxStart]-dT/2,
                    color=col_scheme_BR[0],label=r"$S_{\rm dc}(t)$")
    axs[2,1].set_ylim(-0.3,2.5)
    axs[2,1].set_yticks(ticks=[0,1,2],labels=["0","R","L"])
    axs[2,1].set_xlim(0,time[idxEnd]-time[idxStart]-dT)
    axs[2,1].set_xlabel(r"time $t$ (s)")
    axs[2,1].legend(loc="best")

    axs[2,1].text(0.03, 0.88, "(e)",transform=axs[2,1].transAxes)

    fig.delaxes(axs[2,0])

    plt.tight_layout()
    plt.savefig("figs/paper/endmatter.svg")#,dpi=600)
    plt.show()

    pass

def debug_traces(biases,ids,idx=[0],channel="PCA--",m=3000,k_smooth=2):
    """
    Plot time-dependence of how often the different states appear
    """
    for k, folder_bias in enumerate(biases):
        _ids = np.array(ids[k])[idx]
        for id in _ids:
            states = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy")
            time   = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy")
            M      = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M.npy")

            myDQDClock = DQDClock(1,[0,0],[0,0],[0,0],0)
            myDQDClock.M = M.astype(complex)
            v0 = myDQDClock.getSteadyState()

            where0 = states==0
            where1 = states==1
            where2 = states==2

            where_s = np.array([where0,where1,where2])

            num_states = len(states)

            _tmp = np.linspace(0,(1/2)**(1/3),int(m/10))**3
            gauss = np.concatenate((_tmp[:-1],1-_tmp[::-1],np.ones(m),1-_tmp[:-1],_tmp[::-1]))
            gauss /= np.sum(gauss)
            # gauss=np.ones(m)/m

            plt.figure()
            plt.title(folder_bias+" | "+id)
            color_list = ["red","blue","green"]
            for s in range(3):
                plt.hlines(v0[s],xmin=time[0],xmax = time[num_states-len(gauss)+1],color=color_list[s],label="ss. rates "+str(s))
                plt.hlines(np.average(where_s[s,:]),xmin=time[0],xmax = time[num_states-len(gauss)+1],color=color_list[s],linestyle=":",label="ss. average "+str(s))
                plt.plot(time[0:num_states-len(gauss)+1],np.convolve(where_s[s,:],gauss,mode='valid'),
                            color=color_list[s])
            plt.xlabel("time $t$ (s)")
            plt.ylabel("occupation probability")
            # plt.ylim(-0.1,1.1)
            plt.legend(loc="best")
            plt.show()

            M_list = []
            for k in range(int(num_states/m)):
                try:
                    myRates = RateStatistics(states=states[k*m:(k+1)*m],time=time[:m],identifier=(0,0))
                    M, GammaCond, GammaMarkov, M_err = myRates.time_stats_conditional(plot=False,rates=True,err_analysis=False)
                    M_list.append(M)
                except Exception as error:
                    print(error)

            M_list = np.array(M_list)

            plt.figure()
            plt.title(folder_bias)
            for s in range(3):
                plt.plot(-M_list[:,s,s],label="state "+str(s))
            plt.ylim(np.max(np.abs(M_list))*-0.05,np.max(np.abs(M_list))*1.05)
            plt.show()


def rate_stability_paper(channel="B-V-",sample_width=5000,step=0.2,k_smooth=3):
    """
    Plot time-dependence of how often the different states appear
    """
    biases = ["+0.705 bias","+0.855 bias"]
    bias_labels = ["0.705","0.855"]

    ids = [[id for id in os.listdir('data/'+folder) if os.path.isdir(os.path.join('data/'+folder, id))] for folder in biases]    

    fig, axs = plt.subplots(1,2,figsize=(7.08333,2.0),sharex=True,sharey=True)


    for kDot, folder_bias in enumerate(biases):
        _ids = np.array(ids[kDot])[[0]]
        for kSen, id in enumerate(_ids):
            states = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/states.npy")
            time   = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/time.npy")
            M      = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M.npy")
            M_err  = np.load("data/"+folder_bias+"/"+id+"/"+channel+"K"+str(k_smooth)+"/M_err.npy")

            print(time[sample_width]-time[0])

            num_states  = len(states)
            step_width  = int(sample_width*step)
            num_samples = 1 + int((num_states-sample_width)/step_width)

            diag_array      = np.zeros((num_samples,3))
            diag_err_array  = np.zeros((num_samples,3))
            for k in range(num_samples):
                print(str(k)+" / "+str(num_samples))
                try:
                    myRates = RateStatistics(states=states[k*step_width:k*step_width + sample_width],time=time[:sample_width],identifier=(0,0))
                    _M, GammaCond, GammaMarkov, _M_err = myRates.time_stats_conditional()
                    diag_array[k,:]     = -np.diag(_M)
                    diag_err_array[k,:] = np.abs(np.diag(_M_err))
                except Exception as error:
                    diag_array[k,:]     = diag_array[k-1,:]
                    diag_err_array[k,:] = diag_err_array[k-1,:]
                    print(error)

            colors = ["midnightblue","darkviolet","deeppink"]

            # plt.title(folder_bias)
            for s in range(3):
                # axs[kDot].errorbar(time[:-sample_width:step_width],diag_array[:,s],yerr=diag_err_array[:,s],
                #                    capsize=0,color=colors[s],
                #                    marker=".",markersize=5,label=r"$\Gamma_{"+str(s+1)+r"}$")
                axs[kDot].plot(time[:-sample_width:step_width],diag_array[:,s],color=colors[s],
                                   marker=".",markersize=3,linewidth=0.75,label=r"$\Gamma_{"+str(s+1)+r"}$")
                axs[kDot].hlines(-M[s,s],xmin=0,xmax=time[-sample_width],color=colors[s],
                            linewidth=0.75,linestyle="--")#,label=r"$\Gamma_{"+str(s)+", \mathrm{MLE}}$")
                axs[kDot].fill_between([-1,1+time[-sample_width]],-M[s,s]-M_err[s,s]/2,-M[s,s]+M_err[s,s]/2,color=colors[s],
                            linestyle=":",alpha=0.2)
            
            axs[kDot].set_xlim(0,1700)
            axs[kDot].set_ylabel(r"rate $\Gamma$ (Hz)")
            if kDot==1:
                axs[kDot].legend(loc="upper right")
            # axs[kDot].set_ylim(np.max(np.abs(diag_array))*-0.05,np.max(np.abs(diag_array))*1.05)
            axs[kDot].set_ylim(-3,76)
            axs[kDot].text(0.03, 0.1, r"$V_{\rm DQD}="+bias_labels[kDot]+r"\,{\rm mV}$",transform=axs[kDot].transAxes)
            axs[kDot].set_xlabel(r"lab time $t$ (s)")

    axs[0].text(-0.12, 0.93, "(a)",transform=axs[0].transAxes)
    axs[1].text(-0.09, 0.93, "(b)",transform=axs[1].transAxes)

    plt.tight_layout()
    plt.savefig("figs/paper/rate_stability.pdf")#,dpi=600)
    plt.show()
