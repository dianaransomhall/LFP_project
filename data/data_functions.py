# Function to read in mcd files to python
# import necessary libraries
import neuroshare as ns
import os
import statistics
import pickle
import numpy as np
import shelve
import scipy as sp
import math
import decimal
from scipy import fftpack, signal, stats
from scipy.signal import kaiserord, lfilter, firwin, freqz, butter
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd



#MCDFolderPath = "/Volumes/BACKUP/EXTRAP/FP_SPIKE"
#MCDFilePath = "/Volumes/BACKUP/EXTRAP/FP_SPIKE/ACE/June_04_2014_20613_ACE/060414_20613_ACE_FPSPK.mcd"
#MCDexampleFile = "/Users/dh2744/Dropbox/Documents/Software/Python/LFP_project/nsMCDLibrary_MacOSX_3.7b/ExampleMCD/NeuroshareExample.mcd"


def get_fft( curData, band="alpha"  ):



    want_simple_example = False
    if want_simple_example:

        Fs = 25000
        t = np.asarray(np.arange(0, 1 - (1 / Fs), 1 / Fs))
        s1_temp = lambda t: 0.3 * math.sin(2 * math.pi * 30 * t)
        s2_temp = lambda t: 0.9 * math.sin(2 * math.pi * 45 * t)
        s3_temp = lambda t: 1.2 * math.sin(2 * math.pi * 80  * t)
        s4_temp = lambda t: 1.8 * math.cos(2 * math.pi * 90 * t)

        s1 = np.array([s1_temp(i) for i in t])
        s2 = np.array([s2_temp(i) for i in t])
        s3 = np.array([s3_temp(i) for i in t])
        s4 = np.array([s4_temp(i) for i in t])

        signal = s1 + s2 + s3 +s4
        # this gives which frequencies each vector member belongs to
        W = fftfreq(signal.size, d=t[1] - t[0])
        W = np.array( [round(i) for i in W ] )

        # now we'd like to extract the amplitude and freq of each comp
        y_temp1 = fftpack.fft ( signal )
        ind_30 = np.where(W == 30)
        ind_45 = np.where(W==45) # to get specific frequency bands, 45 Hz
        ind_80 = np.where(W == 80)
        ind_90 = np.where(W == 90)

        np.real(y_temp1[ind_30])
        np.real(y_temp1[ind_45])
        np.real(y_temp1[ind_80])
        np.real(y_temp1[ind_90])

        # magnitude
        mag = math.sqrt(np.real(y_temp1[ind_30])**2 + np.imag(y_temp1[ind_30])**2 )
        N=Fs
        plt.plot(np.abs(y_temp1[:150])  ) # this does give is correct frequencies
        y_ifft = fftpack.ifft(y_temp1) # this gives us back full signal
        plt.plot(y_ifft)
        # freq 30 wave
        y_ifft2 = fftpack.ifft(y_temp1[0:32])
        plt.plot(y_ifft2)
        # freq 45 wave
        y_ifft3 = fftpack.ifft(y_temp1[32:46])
        plt.plot(y_ifft3)
        # freq 80 wave
        y_ifft4 = fftpack.ifft(y_temp1[46:85])
        plt.plot(y_ifft4)
        # freq 90 wave
        y_ifft5 = fftpack.ifft(y_temp1[85:95])
        plt.plot(y_ifft5)
    #end of simple example

    ind_delta = [1,2,3, 4 ]
    ind_theta = [5,6,7, 8]
    ind_alpha = [9,10,11,12,13]
    ind_beta = np.array(range(14,30))
    ind_gamma = np.array(range(31, 50))

    if band=="delta":
        ind_want=ind_delta
    elif band=="theta":
        ind_want = ind_theta
    elif band=="alpha":
        ind_want = ind_alpha
    elif band=="beta":
        ind_want = ind_beta
    elif band=="gamma":
        ind_want = ind_gamma


    #plot raw signal
    want_raw_signal_plot=False
    if want_raw_signal_plot:
        plt.plot( curData )
        plt.ylabel(' raw ')
        plt.xlabel('1/25,000 second ')
        # plt.axis( [ 0, 200, 0, max(abs( ifft_ind ) ) ] )
        plt.show()


    #do fft

    #August 10, 2019, down sample 25k/sec to 1k/sec

    # run fft on data applied to kaiser window
    fft_resp = fftpack.fft(curData)
    # make a set of indicators for band of interest
    ind = np.zeros(len(fft_resp))
    ind[ind_want] = 1

    power_want = np.mean( abs( fft_resp[ind_want] ) )

    want_band_plot=False
    if want_band_plot:

        ifft_ind = fftpack.ifft( fft_resp*ind )
        plt.plot(ifft_ind )
        plt.plot(curData, alpha=0.2)
        plt.ylabel(' power ')
        plt.xlabel('1/25,000 second ')
        #plt.axis( [ 0, 200, 0, max(abs( ifft_ind ) ) ] )
        plt.show()

    return power_want

def get_chNamesList(fd, LFP=True):

    chNamesList=list()
    if LFP:
        for i in range(60, 120):
            # i=60
            print(fd.entities[i].label[25:])
            chNamesList.append(fd.entities[i].label[25:])
    else:
        for i in range(0, 60):
            # i=60
            print(fd.entities[i].label[25:])
            chNamesList.append(fd.entities[i].label[25:])

    return chNamesList


def get_spikes(chem):
        chem_dict = dict()
        files = get_files(chem)
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            fd = ns.File(f)
            chNamesList = get_chNamesList(fd)
            spikes = dict()

            # fd.entities[0].get_data(200) gives you the 200th spike and the LFP or shape of spike,
            for numCh in range(0, 60):

                cur_entity = fd.entities[numCh]
                cur_spk_train = []
                for curSp in range(0, cur_entity.item_count):
                    # curSp = 0
                    cur_spk_train.append(cur_entity.get_data(curSp)[1])

                spikes[chNamesList[numCh]] = cur_spk_train

            chem_dict[title] = spikes
        return chem_dict
# end of get_spikes


def get_spikes_wrapper(chem="bic", save=True):
    #dom_spk.keys()
    #dom_spk['20608_DA_FPSPK'].keys()
    # dom_spk['20608_DA_FPSPK']['14']
    # get

    if chem=="bic":
        bic_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_bic_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(bic_spk, f)
    elif chem=="dom":
        dom_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dom_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(dom_spk, f)
    elif chem=="gly":
        gly_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_gly_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(gly_spk, f)
    elif chem=="ace":
        ace_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_ace_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(ace_spk, f)
    elif chem=="h2o":
        h2o_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_h2o_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(h2o_spk, f)
    elif chem=="dmso":
        dmso_spk = get_spikes(chem)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dmso_spk.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(dmso_spk, f)

    # end of get_spikes_wrapper




def MCD_read(MCDFilePath):


    # open file using the neuroshare bindings
    fd = ns.File(MCDFilePath)

    #entity_type: 3 is spikes, 2 in analogue

    for i, entity in enumerate(fd.entities):
        print((i, entity.label, entity.entity_type))

    # (108, 'filt0001 0059 0048       57', 2)
    # 0059 is the number of the corresponding channel in spk, 0048 is the number in analogue
    # 57 is name of channel in matrix notation, 8*8 - 4 on corner channels
    # use matrix notation to spk to lfp

    #create empty dictionary
    data = dict()

    numCh = 60
    analog1 = fd.entities[numCh]  # open analog signal entity
    print(analog1.units)  #V
    print(analog1.sample_rate ) # 25,000/second

    temp_data_fft = dict()

    #get E names
    chNamesList=get_chNamesList(fd, LFP='True' )




    fft_byE = pd.DataFrame(0,
                         index=chNamesList,
                         columns=('delta', 'theta', 'alpha', 'beta', 'gamma'))

    chNamesList_spikes=get_chNamesList(fd, LFP='False' )
    #spikes=get_spikes(fd, chNamesList_spikes)


    for numCh in range(60,120):

        print("numCh is "+ str(numCh)  )
        analog1 = fd.entities[numCh]  # open analog signal entity

        entity =  fd.entities[ numCh ]
        print(fd.entities[ numCh ].label , entity.entity_type )
        # spikes is 0-59
        # filt0001 is 60 on
        # len(fd.entities) 120

        data1, times, count = analog1.get_data()
        # count 7,657,500 is number of samples
        # times a numeric array of when samples took place (s)
        # analog1.entity_type

        # create channel names
        channelName = entity.label[0:4] + entity.label[23:]
        channelNum = channelName.split(" ")[2]

        # store data with name in the dictionary
        data2 = np.array(data1)

        temp_data_fft = np.zeros(shape=(math.floor(max(times)), 50))
        sec = 1
        totalSec = math.floor(max(times))

        # August 10, 2019 downsample to 1k/sec from 25k/sec
        data3 = data2[0:(totalSec*25000)] # remove tail partial second

        # August 10, 2019 low pass FIR filter to eliminate aliasing
        sample_rate = 25000
        # The Nyquist rate of the signal.
        nyq_rate = sample_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 5.0 / nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 200.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # Use lfilter to filter x with the FIR filter.
        data4 = lfilter(taps, 1.0, data3)


        #down sample to 1000 samples/sec
        data[channelName] = signal.resample(data4,
                                        num=(1000*totalSec),
                                        t=None,
                                        axis=0,
                                        window=None)


        #make an empty data frame
        fft_r = pd.DataFrame( 0,
                           index=np.arange(totalSec )  ,
                           columns=('delta', 'theta', 'alpha', 'beta', 'gamma'))
        # fft_r.loc[:,"alpha" ], fft_r.loc[1,:]

        iterations = np.arange(2, totalSec, 0.5)
        for sec in iterations:

            fs = 1000;


            #August 10, 2019: move along the signal in 0.5s increments
            # take 2 full seconds of data
            start_signal = int((sec-1.5)*fs)
            end_signal = int((sec+0.5)*fs)
            curData_temp = data[channelName][start_signal:end_signal ]

            beta = 0.5  # default in matlab documentation
            w_kaiser = signal.get_window(window=('kaiser', beta), Nx=2*fs, fftbins=False)

            curData = w_kaiser * curData_temp; # element wise operation

            #band pass filter
            order=2000 #order of filter is same as number of obs that go into filter
            def butter_bandpass(lowcut, highcut, fs, order=order):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a

            def butter_bandpass_filter(data, lowcut, highcut, fs, order=order):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = lfilter(b, a, data)
                return y

            #sample rate and desired cutoff frequencies in Hz
            lowcut = 1
            highcut = 4
            y = butter_bandpass_filter(curData, lowcut = lowcut,
                                       highcut = highcut, fs=1000 )

            band_want = "delta"
            power_delta = get_fft(curData, band_want)
            fft_r.loc[sec, "delta"] = power_delta

            band_want = "theta"
            power_theta = get_fft(curData, band_want)
            fft_r.loc[sec, "theta"] = power_theta

            band_want = "alpha"
            power_alpha = get_fft(curData, band_want)
            fft_r.loc[sec, "alpha"] = power_alpha

            band_want = "beta"
            power_beta = get_fft(curData, band_want)
            fft_r.loc[sec, "beta"] = power_beta

            band_want = "gamma"
            power_gamma = get_fft(curData, band_want)
            fft_r.loc[sec, "gamma"] = power_gamma

        #end of for loop

        # do averaging across all seconds for each band put in the right electrode
        fft_byE.loc[channelNum, :] =fft_r.mean(axis=0)



    # end of loop through Channels

    return fft_byE



def get_files(chem="bic"):

    if chem=="bic":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/BIC'
    elif chem=="ace":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/ACE'
    elif chem=="gly":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/GLY'
    elif chem=="dmso":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/DMSO'
    elif chem=="dom":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/DOM'
    elif chem=="h2o":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/H2O'
    elif chem=="per25":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/PER25'
    elif chem=="per50":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/PER50'
    elif chem=="lin1":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/LIN1'
    elif chem=="lin10":
        path = '/Volumes/BACKUP/EXTRAP/FP_SPIKE/LIN10'

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.mcd' in file:
                files.append(os.path.join(r, file))

    return files






def run_one_chemical(chem="bic", save="True"):

    if chem=="bic":
        bic = dict()
        files = get_files("bic" )
        for f in files:
            print(f)
            title = '_'.join( ( f.split(".mcd")[0].split("_")[  (len(f.split(".mcd")[0].split("_"))-3):] ) )
            bic[title]=MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_bic.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(bic, f)
        return bic
    elif chem=="ace":
        ace = dict()
        files = get_files("ace")
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            ace[title] = MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_ace.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(ace, f)
        return ace
    elif chem=="h2o":
        h2o = dict()
        files = get_files("h2o")
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            h2o[title] = MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_h2o.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(h2o, f)
        return h2o
    elif chem=="gly":
        gly=dict()
        files = get_files("gly")
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            gly[title] = MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_gly.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(gly, f)
        return gly
    elif chem=="dmso":
        dmso=dict()
        files = get_files("dmso")
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            dmso[title] = MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dmso.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(dmso, f)
        return dmso
    elif chem == "dom":
        dom=dict()
        files = get_files("dom")
        for f in files:
            print(f)
            title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
            dom[title] = MCD_read(f)
        if save:
            filename="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dom.pkl"
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(dom, f)
        return dom


