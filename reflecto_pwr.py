import numpy as np
import os
import matplotlib.pyplot as plt

from data_loader import load_id

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Script for converting time traces in volts to power [Watts].
# Can then be used for calculating power dissipation via reflectometry measurements.
# Files manually loaded from adding to directory.
# Only setting that needs changing is LO_FREQ to match the frequency used for measurement.
# LO_FREQ must only take on values specified.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data is stored in volts and seconds

INPUT_SIGNAL_POWER = -20 # in dBm

########################################

# Attenuators & amplifiers
dircoupler_att = -20 #dB (directional coupler)
tot_fridge_att = -31 #dB (attenuators in fridge)
rt_att = -20 #dB (attenuator placed outside fridge)
cryo_amp = 32 #dB (cryogenic amplifier)

# Power into sample (from attenuators)
power_in_sample = INPUT_SIGNAL_POWER+rt_att+tot_fridge_att+dircoupler_att # in dBm


###############################
# Power conversion functions ##
def dBm_to_vpp(power):
    vpp = np.sqrt(100*(10**(power/10))/1000)
    return vpp

def dBm_to_watts(PdBm):
    return (10**(PdBm/10)) / 1000

def vpp_to_vrms(vpp):
    vrms = vpp/2*np.sqrt(2)
    return vrms

def vrms_to_vpp(Vrms):
    vpp = Vrms*np.sqrt(2)/2
    return vpp

# voltage output from attenuators functions #
def v_out_component(v_in, gain):
    v_out = v_in*10**(gain/20) # Here gain refers to the attenuation of component
    return v_out

def v_in_component(v_out, gain):
    # For calculating the effect of amplifier in Rx line
    v_in = v_out/(10**(gain/20))
    return v_in

###############################

def get_reflectometry_power(folder_bias,id):
    """
    Returns average dissipated power (in Watts) for a given time-trace

    Input:
        folder_bias, id --> bias (DQD) and id (sensor setting)
    Output:
        avg_pwr
    """
    ##################
    # Initialization #
    ##################
    path = 'data/'+folder_bias+'/'+id # Change to directory with data
    
    # File names for the data folder
    files_X = ['dummy_parameter--','C-V-','time-s-']
    files_Y = ['dummy_parameter--','D-V-','time-s-']

    # Obtain the measurement traces
    demod_X = load_id("data/"+folder_bias,id,data_format=None,file_names=files_X)[0]
    demod_Y, dpndnt_var, meta_data = load_id("data/"+folder_bias,id,data_format=None,file_names=files_Y)

    time = dpndnt_var - dpndnt_var[0]

    ###########################
    # Setting of LO Frequency #
    ###########################
    dac2 = meta_data['instrument_summary']['dac']['dac2']

    if np.allclose(dac2,0.102):
        LO_FREQ = 128e6
    elif np.allclose(dac2,0.183):
        LO_FREQ = 116e6
    elif np.allclose(dac2,0.264):
        LO_FREQ = 115e6
    elif np.allclose(dac2,0.345):
        LO_FREQ = 114e6
    else:
        print("Error in determining sensor SNR for reflectometry power measurement")
        print("Using standard value.")

    # LO_FREQ = 114e6 # Use either 114e6, 115e6, 116e6 or 128e6

    ###############################

    if LO_FREQ==114e6:
        V_OFFSET_X = 2.213
        V_PREOFFSET_X = -94.11434960e-3
        V_SCALE_X = 10

        V_OFFSET_Y = 9.208
        V_PREOFFSET_Y = -747.83410401e-3
        V_SCALE_Y = 10

        INPUT_SCALING = 1e3

    elif LO_FREQ==115e6:
        V_OFFSET_X = 1.231
        V_PREOFFSET_X = 71.15787242e-3
        V_SCALE_X = 10

        V_OFFSET_Y = 9.814
        V_PREOFFSET_Y = -898.73040311e-3
        V_SCALE_Y = 10

        INPUT_SCALING = 1e3

    elif LO_FREQ==116e6:
        V_OFFSET_X = 1.006
        V_PREOFFSET_X = 87.42342046e-3
        V_SCALE_X = 10

        V_OFFSET_Y = 9.868
        V_PREOFFSET_Y = -1.00398345
        V_SCALE_Y = 10

        INPUT_SCALING = 1e3

    elif LO_FREQ==128e6:
        V_OFFSET_X = 2.408
        V_PREOFFSET_X = -277.55189046e-3
        V_SCALE_X = 10

        V_OFFSET_Y = 9.344
        V_PREOFFSET_Y = -982.78429983e-3
        V_SCALE_Y = 10

        INPUT_SCALING = 1e3


    #####################
    # Power calculation #
    #####################
    ##### Plotting Offsets ##########
    X_unscaled = (demod_X-V_OFFSET_X/V_SCALE_X)-V_PREOFFSET_X
    Y_unscaled = (demod_Y-V_OFFSET_Y/V_SCALE_Y)-V_PREOFFSET_Y

    X_unscaled = X_unscaled/INPUT_SCALING
    Y_unscaled = Y_unscaled/INPUT_SCALING

    ########### R value ###########
    phi = np.arctan2(Y_unscaled, X_unscaled) # Phase difference between ref and input signal to lock-in

    R_rms = np.sqrt((X_unscaled/np.cos(phi))**2+(Y_unscaled/np.sin(phi))**2) # Voltage into lock-in rms
    R_pp = vrms_to_vpp(R_rms) # Voltage into lock-in peak-to-peak


    #### Voltage out of sample ###
    v_out_sample_pp = v_in_component(R_pp, cryo_amp) # Reversing effect of +32dB cryogenic amplifier
    v_out_sample_rms = vpp_to_vrms(v_out_sample_pp)


    #### Voltage into sample #####
    v_into_fridge_pp = dBm_to_vpp(INPUT_SIGNAL_POWER)
    v_into_fridge = (v_into_fridge_pp/2)*np.sin(2*np.pi*LO_FREQ*time) # Waveform of input signal

    v_atten_rt = v_out_component(v_into_fridge, rt_att) # room temperature attenuator outside fridge
    v_atten_fridge = v_out_component(v_atten_rt, tot_fridge_att) # Fridge
    v_atten_dir_coup = v_out_component(v_atten_fridge, dircoupler_att) # directional coupler

    v_into_sample = v_atten_dir_coup

    v_into_sample_pp = np.max(v_into_sample)-np.min(v_into_sample)
    v_into_sample_rms = vpp_to_vrms(v_into_sample_pp)


    #### Power output calculation ####
    power_out = 20*np.log(v_out_sample_rms/v_into_sample_rms)+power_in_sample # gain across sample + power into sample (in dBm)
    power_out_Watts = dBm_to_watts(power_out)

    return np.average(power_out_Watts)

    #### Plotting ####
    plt.figure()
    plt.plot(time[0:500], power_out_Watts[0:500])
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('power [Watts]')
    plt.show()