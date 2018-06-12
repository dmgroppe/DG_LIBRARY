import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy import signal
from scipy import stats
import scipy.stats
import mtspec
import dgFuncs as dg
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_path_dict():
    """ Returns a dictionary of project folders. Folders vary by machine.
    Makes it easy to change folders."""
    path_dict=dict()
    if sys.platform=='linux':
        path_dict['ieeg_root']='/home/dgroppe/TWH_EEG'
        path_dict['eu_root'] = '/home/dgroppe/EU_MAT_DATA'
        #path_dict['eu_meta'] = '/home/dgroppe/EU_METADATA/'
        path_dict['eu_meta'] = '/home/dgroppe/GIT/SZR_ANT/EU_METADATA/'
        path_dict['pics']='/home/dgroppe/GIT/SZR_ANT/PICS/' #TODO fix this
        path_dict['onset_csv'] = '/home/dgroppe/TWH_INFO/CLINICIAN_ONSET_TIMES'
        path_dict['ftrs_root'] = '/home/dgroppe/GIT/SZR_ANT/FTRS'
        path_dict['szr_ant_root'] = '/home/dgroppe/GIT/SZR_ANT/'
        path_dict['eu_gen_ftrs']='/home/dgroppe/GIT/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/'
    else:
        #path_dict['ieeg_root']='/Users/davidgroppe/ONGOING/SZR_SPREAD/PATIENTS/'
        path_dict['ieeg_root'] = '/Users/davidgroppe/ONGOING/TWH_EEG/'
        path_dict['eu_root'] = '/Users/davidgroppe/ONGOING/EU_EEG/'
        #path_dict['eu_meta'] = '/Users/davidgroppe/Dropbox/TWH_INFO/EU_METADATA/'
        path_dict['eu_meta'] = '/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/'
        path_dict['pics']='/Users/davidgroppe/PycharmProjects/SZR_ANT/PICS/'
        path_dict['onset_csv'] = '/Users/davidgroppe/Dropbox/TWH_INFO/CLINICIAN_ONSET_TIMES'
        path_dict['ftrs_root'] = '/Users/davidgroppe/PycharmProjects/SZR_ANT/FTRS'
        path_dict['szr_ant_root'] = '/Users/davidgroppe/PycharmProjects/SZR_ANT/'
        path_dict['eu_gen_ftrs'] = '/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/'
    return path_dict


def clean_labels(raw_labels):
    cln_labels=[]
    for lab in raw_labels:
        half1, half2=lab.split('-')
        new_label=[]
        new_label.append(half1)
        new_label.append('-')
        for c in half2:
            if c.isdigit():
                new_label.append(c)
        cln_labels.append(''.join(new_label))
    return cln_labels


def strat_plot(dat,chan_labels,fig_id=1,show_chan_ids=None,h_offset=2,srate=1,tpts_sec=None,fontsize=9):
    if show_chan_ids is None:
        show_chan_ids=np.arange(len(chan_labels))
    # print('dat shape={}'.format(dat.shape))
    # print('len(chan_labels)=%d' % len(chan_labels))
    temp_dat=dat[show_chan_ids,:].copy()
    n_show_chan=len(show_chan_ids)
    n_tpt=temp_dat.shape[1]
    if tpts_sec is None:
        tpts_sec=np.arange(n_tpt)/srate
    for a in range(n_show_chan):
        temp_dat[a,:]=temp_dat[a,:]-np.mean(temp_dat[a,:])+a*h_offset
    mn_y=np.min(temp_dat[0,:])
    mx_y=np.max(temp_dat[-1,:])
    h=plt.figure(fig_id)
    plt.clf()
    plt.plot(tpts_sec,temp_dat.T)
    plt.xlim(tpts_sec[0],tpts_sec[-1])
    plt.ylim(mn_y,mx_y)
    plt.xlabel('Seconds')
    plt.yticks(np.arange(n_show_chan)*h_offset)
    ax=h.axes[0]
    temp_chan_labels=np.array(chan_labels)
    ax.set_yticklabels(temp_chan_labels[show_chan_ids],fontsize=fontsize)
    return h, ax


def causal_butter(data,srate,passband,filt_order):
    """ Applies a causal, digital band/hi/lowpass Butterworth filter to a matrix of data.
    Inputs:
      data: time x channel
      srate: sampling rate
      pass band: tuple of low and high passband boundaries
      filt_order: order of the butterworth filter"""
    from scipy import signal
    n_chan, n_tpt = data.shape
    # print('chans %d' % n_chan)
    # print('tpts %d' % n_tpt)
    Nyq=srate/2

    if passband[0]==0:
        b, a = signal.butter(filt_order, passband[1]/Nyq, 'lowpass', analog=False)
    elif passband[1]==Nyq:
        b, a = signal.butter(filt_order, passband[0] / Nyq, 'highpass', analog=False)
    else:
        b, a = signal.butter(filt_order, [passband[0]/Nyq, passband[1]/Nyq], 'bandpass', analog=False)

    filtered_data=signal.lfilter(b,a,data)
    return filtered_data


def sgram_plot(sgram,sgram_sec,title=None,fname=None,fig_id=1,onset_lower_bnd_sec=None,onset_upper_bnd_sec=None):
    """ plots spectrogram-like plots at one channel and colorbar
    +/- abs_mx colorscale is assumed
    """
    plt.figure(fig_id)
    plt.clf()
    ax = plt.gca()
    abs_mx = np.max(np.abs(sgram))
    im = ax.imshow(sgram, vmin=-abs_mx, vmax=abs_mx)
    ylim = plt.ylim()
    if onset_lower_bnd_sec!=None:
        # Plot vert lines to indicate szr onset window
        onset_sgram_tpt_lower = dg.find_nearest(sgram_sec, onset_lower_bnd_sec)
        onset_sgram_tpt_upper = dg.find_nearest(sgram_sec, onset_upper_bnd_sec)
        plt.plot([onset_sgram_tpt_upper, onset_sgram_tpt_upper], ylim, 'k--')
        plt.plot([onset_sgram_tpt_lower, onset_sgram_tpt_lower], ylim, 'k--')
    plt.ylim(ylim)
    plt.ylabel('Hz')
    raw_xticks = plt.xticks()
    xtick_labels = list()
    n_wind=len(sgram_sec)
    for tick in raw_xticks[0]:
        if tick < n_wind:
            xtick_labels.append(str(int(sgram_sec[int(tick)])))
        else:
            xtick_labels.append('noData')
    _ = plt.xticks(raw_xticks[0], xtick_labels)  # works
    plt.xlim([0, len(sgram_sec)])
    plt.xlabel('Seconds')
    plt.gca().invert_yaxis()
    if title!=None:
        plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Rounds to 1 postdecimal digit
    cbar_max_tick = int(np.floor(abs_mx * 10)) / 10
    cbar_min_tick = -cbar_max_tick
    cbar = plt.colorbar(im, cax=cax, ticks=[cbar_min_tick, 0, cbar_max_tick])
    if fname!=None:
        plt.savefig(fname)

def bp_pwr(data, srate, wind_len, wind_step, n_tapers, tpts_sec, bands, taper='slepian'):
    """ Returns the mean power (in dB) of data in bands
     Input:
       data - needs to be a 2D numpy matrix (not a 1D array)"""
    n_chan, n_tpt = data.shape
    print('chans %d' % n_chan)
    print('tpts %d' % n_tpt)
    n_band=len(bands)
    use_ids=dict()
    # ToDo: ignore DC and harmonics
    for chan in range(n_chan):
        if taper=='slepian':
            sgram, f, sgram_sec = mt_sgram(data[chan, :], srate, wind_len, wind_step, n_tapers, tpts_sec)
        else:
            sgram, f, sgram_sec = hamming_sgram(data[chan, :], srate, wind_len, wind_step, tpts_sec)
        if chan == 0:
            n_wind = len(sgram_sec)
            db_pwr = np.zeros((n_chan, n_band, n_wind))
            # figure out which freqs to use
            for b in bands:
                use_ids[b] = np.logical_and(f > b[0], f < b[1])
        for b_ct, b in enumerate(bands):
            db_pwr[chan, b_ct, :] = np.mean(sgram[use_ids[b], :], axis=0)
    return db_pwr, sgram_sec


def bp_hilb_phz_dif(data, Sf, wind_len, wind_step, tpts_sec, bands, filt_order=4):
    """ Returns the mean of the difference between the Hilbert transform of channel pairs using a moving window after
    bandpass filtering with a causal Butterworth filter. This a complex valued time series. The magnitude of this is
    the phase locking value (PLV), in each moving window. You average adjacent values to derive PLV values across
    longer time values.

    NOTE THAT THE FIRST ROW OF data IS ASSUMED TO BE THE SEED CHANNEL. The Hilbert trans-difference between the seed
    channel and all other channels is returned.

     Input:
       data - needs to be a 2D numpy matrix (channel x time) not a 1D array
       Sf - sampling rate in Hz
       wind_len - window length in time points
       wind_step - window step size in time points
       tpts_sec - time of each time point in seconds
       bands - list of tuples indicating frequency bands
       filt_order - order of the Butterworth filter (default=4)
       """
    n_chan, n_tpt = data.shape
    # print('chans %d' % n_chan)
    # print('tpts %d' % n_tpt)
    n_band=len(bands)

    n_half_wind = int(np.round(wind_len / 2))
    n_hilb_tpt = len(np.arange(n_half_wind, n_tpt - n_half_wind, wind_step))
    hilb_phz_dif = np.zeros((n_band, n_chan-1, n_hilb_tpt),dtype='complex_')
    hilb_sec = np.zeros(n_hilb_tpt)
    # Bandpass filter the data
    for band_ct, band in enumerate(bands):
        # print('Working on {} Hz'.format(band))
        bp_data = causal_butter(data, Sf, band, filt_order)
        # Moving window
        hilb_ct = 0
        for tpt_ct in range(n_half_wind, n_tpt - n_half_wind, wind_step):
            hilb = signal.hilbert(bp_data[:, (tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])

            for chan_ct in range(1,n_chan):
                hilb_diff=hilb[0,:]-hilb[chan_ct,:]
                hilb_diff_nrm=np.divide(hilb_diff,np.abs(hilb_diff))
                hilb_phz_dif[band_ct,chan_ct-1,hilb_ct]=np.mean(hilb_diff_nrm)
            if band_ct==0:
                # only need to do this for first frequency band
                hilb_sec[hilb_ct] = np.mean(tpts_sec[(tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
            hilb_ct += 1

    if n_chan==2:
        hilb_phz_dif=np.squeeze(hilb_phz_dif)

    return hilb_phz_dif, hilb_sec


def bp_hilb_phz_dif_delta(data, Sf, wind_len, wind_step, tpts_sec, bands, filt_order=4):
    """ Returns the mean of the difference between the Hilbert transform of channel pairs using a moving window after
    bandpass filtering with a causal Butterworth filter. This a complex valued time series. The magnitude of this is
    the phase locking value (PLV), in each moving window. You average adjacent values to derive PLV values across
    longer time values.

    NOTE THAT THE FIRST ROW OF data IS ASSUMED TO BE THE SEED CHANNEL. The Hilbert trans-difference between the seed
    channel and all other channels is returned.

     Input:
       data - needs to be a 2D numpy matrix (channel x time) not a 1D array
       Sf - sampling rate in Hz
       wind_len - window length in time points
       wind_step - window step size in time points
       tpts_sec - time of each time point in seconds
       bands - list of tuples indicating frequency bands
       filt_order - order of the Butterworth filter (default=4)
       """
    n_chan, n_tpt = data.shape
    # print('chans %d' % n_chan)
    # print('tpts %d' % n_tpt)
    n_band=len(bands)

    n_half_wind = int(np.round(wind_len / 2))
    n_hilb_tpt = len(np.arange(n_half_wind, n_tpt - n_half_wind, wind_step))
    hilb_ang_dif = np.zeros((n_band, n_chan-1, n_hilb_tpt),dtype='complex_')
    hilb_sec = np.zeros(n_hilb_tpt)
    # Bandpass filter the data
    for band_ct, band in enumerate(bands):
        # print('Working on {} Hz'.format(band))
        bp_data = causal_butter(data, Sf, band, filt_order)
        # Moving window
        hilb_ct = 0
        for tpt_ct in range(n_half_wind, n_tpt - n_half_wind, wind_step):
            hilb = signal.hilbert(bp_data[:, (tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
            for chan_ct in range(1,n_chan):
                hilb_diff=hilb[0,:]-hilb[chan_ct,:]
                hilb_ang = np.angle(hilb_diff)  # This ranges from -pi to pi
                hilb_ang = np.unwrap(hilb_ang)
                hilb_ang_dif[band_ct,chan_ct-1,hilb_ct]=np.mean(np.abs(np.diff(hilb_ang)))
            if band_ct==0:
                # only need to do this for first frequency band
                hilb_sec[hilb_ct] = np.mean(tpts_sec[(tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
            hilb_ct += 1

    if n_chan==2:
        hilb_ang_dif=np.squeeze(hilb_ang_dif)

    return hilb_ang_dif, hilb_sec


def bp_hilb_mag(data, Sf, wind_len, wind_step, tpts_sec, bands, filt_order=4):
    """ Returns the abs value of the moving window hilbert transform of data in bands after bandpass filtering
    with a causal Butterworth filter
     Input:
       data - needs to be a 2D numpy matrix (not a 1D array)
       Sf - sampling rate in Hz
       wind_len - window length in time points
       wind_step - window step size in time points
       tpts_sec - time of each time point in seconds
       bands - list of tuples indicating frequency bands
       filt_order - order of the Butterworth filter (default=4)
       """
    n_chan, n_tpt = data.shape
    print('chans %d' % n_chan)
    print('tpts %d' % n_tpt)
    n_band=len(bands)

    n_half_wind = int(np.round(wind_len / 2))
    n_hilby_tpt = len(np.arange(n_half_wind, n_tpt - n_half_wind, wind_step))
    hilb_mag = np.zeros((n_band, n_chan, n_hilby_tpt))
    hilb_inst_freq = np.zeros((n_band, n_chan, n_hilby_tpt))
    hilb_delt_freq = np.zeros((n_band, n_chan, n_hilby_tpt))
    hilb_sec = np.zeros(n_hilby_tpt)
    # Bandpass filter the data
    for band_ct, band in enumerate(bands):
        print('Working on {} Hz'.format(band))
        bp_data = causal_butter(data, Sf, band, filt_order)
        # Moving window
        hilb_ct = 0
        for tpt_ct in range(n_half_wind, n_tpt - n_half_wind, wind_step):
            hilby = signal.hilbert(bp_data[:, (tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
            #print(hilby.shape)
            hilb_mag[band_ct, :, hilb_ct] = np.mean(np.abs(hilby), axis=1)
            hilby_clip_phz = np.unwrap(np.angle(hilby))
            clip_instant_freq = (np.diff(hilby_clip_phz) / (2.0 * np.pi) * Sf)
            hilb_inst_freq[band_ct, :, hilb_ct]=np.mean(clip_instant_freq)
            #hilb_delt_freq[band_ct, :, hilb_ct] = np.mean(np.diff(clip_instant_freq)) # mean change in inst. freq
            #t-score hilb_delt_freq[band_ct, :, hilb_ct]=np.mean(np.diff(clip_instant_freq))/np.std(np.diff(clip_instant_freq))
            if band_ct==0:
                # only need to do this for first frequency band
                hilb_sec[hilb_ct] = np.mean(tpts_sec[(tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
            hilb_ct += 1
    if n_chan==1:
        hilb_mag=np.squeeze(hilb_mag)
        hilb_inst_freq=np.squeeze(hilb_inst_freq)
        hilb_delt_freq=np.squeeze(hilb_delt_freq)

    return hilb_mag, hilb_inst_freq, hilb_sec


def band_coh(x, y, N, f_band_ids):
    """Computes the coherence between time series with a list of frequency bands
    Inputs:
      x, y - two 1D time series of equal length
      Sf - sampling rate
      N - # of time points to use for FFT. If N># of time points, 0 padding will be used. If N<# of time points, data is cropped.
      f_band_ids - list of tuples indicating the *index* of the lower and upper bound of each frequency band.

    Outputs:
      coh - vector of len(f_band_ids) with the coherence of each frequency band. Possible values range from 0 to 1.
      """

    dft_x = np.fft.rfft(x, N)
    dft_y = np.fft.rfft(y, N)

    xx = np.multiply(x, np.conj(x)) / N
    yy = np.multiply(y, np.conj(y)) / N
    xy = np.multiply(x, np.conj(y)) / N

    n_band = len(f_band_ids)
    coh = np.zeros(n_band)
    for a in range(n_band):
        mn_xx = np.mean(xx[f_band_ids[a][0]:f_band_ids[a][1]])
        mn_yy = np.mean(yy[f_band_ids[a][0]:f_band_ids[a][1]])
        mn_xy = np.mean(xy[f_band_ids[a][0]:f_band_ids[a][1]])
        coh[a] = (np.abs(mn_xy) ** 2) / (mn_xx * mn_yy)

    return coh


def bp_coh(data1, data2, srate, wind_len, wind_step, tpts_sec, bands):
    """ This uses custom FFT-based code that calculates the coherence between
    channels in a one second moving window. It is different from standard
    approaches in that it does not require multiple trials or tapers.

    Inputs:
     data1 - 1D time series
     data2 - 1D time series
     srate - Sampling rate in Hz
     wind_len - length of moving time window in time points (NOT seconds)
     wind_step - moving time window step in units of time points (NOT seconds)
     tpts_sec - 1D vector of times (in seconds) of each time point
     bands - list of 2D tuples that define lower and upper boundaries of frequency bands in Hz
    """
    # ToDo: ignore line noise and harmonics?
    n_band=len(bands)
    n_tpt = len(data1)

    if len(data2)!=n_tpt:
        print('Error: data1 and data2 are not of equal lenght.')
        coh=None
        coh_sec=None
    else:
        wind_cntr = np.arange(wind_len, n_tpt, wind_step)
        wind_len = int(np.round(wind_len))
        wind_step = int(np.round(wind_step))

        # Compute # of time windows
        n_wind = len(wind_cntr)

        # Compute frequencies
        nyq=srate/2
        T=wind_len/srate # length of moving window in seconds
        freq_spacing = 1 / T
        f = np.linspace(0, nyq, 1 + nyq / freq_spacing)

        f_band_ids=list()
        for ct, band in enumerate(bands):
            low_bnd_id=dg.find_nearest(f,band[0])
            up_bnd_id = dg.find_nearest(f, band[1])
            f_band_ids.append((low_bnd_id,up_bnd_id))
            # Code below for error checking
            # print('{}: {} to {} Hz, {} to {} ID'.format(ct,f[low_bnd_id],f[up_bnd_id],
            #                                         low_bnd_id,up_bnd_id))

        taper = np.hamming(wind_len)

        cursor = 0
        coh=np.zeros((n_band,n_wind))
        coh_sec = np.zeros(n_wind)
        for wind in range(n_wind):
            wind_coh=band_coh(np.multiply(taper,data1[cursor:cursor + wind_len]),
                          np.multiply(taper,data2[cursor:cursor + wind_len]), wind_len, f_band_ids)
            coh[:, wind] = wind_coh
            coh_sec[wind] = np.mean(tpts_sec[cursor:cursor + wind_len])
            cursor += wind_step

    return coh, coh_sec


def omni_coh(data, seed_chan_id, N, n_freq):
    """ Computes the coherence between a seed channel and all other channels at all frequencies.
    One coherence value per frequency is returned.

    Inputs:
    data - channel x time matrix
    seed_chan_id - index of seed channel
    N - # of time points in DFT window (bigger values lead to padding, smaller cropping)
    n_freq - # of frequences that will be returned by the DFT
      """

    n_chan = data.shape[0]

    # Compute hamming taper
    taper = np.hamming(N)

    # Compute FFT for seed chan
    dft_x = np.fft.rfft(np.multiply(data[seed_chan_id, :], taper), N)
    xx = np.multiply(dft_x, np.conj(dft_x)) / N

    omni_yy = np.zeros((n_freq, n_chan - 1), dtype=complex)
    omni_xy = np.zeros((n_freq, n_chan - 1), dtype=complex)

    ct = 0
    for chan_loop in range(n_chan):
        if chan_loop != seed_chan_id:
            dft_y = np.fft.rfft(np.multiply(data[chan_loop, :], taper), N)
            omni_yy[:, ct] = np.multiply(dft_y, np.conj(dft_y)) / N
            omni_xy[:, ct] = np.multiply(dft_x, np.conj(dft_y)) / N
            ct += 1

    mn_yy = np.mean(omni_yy, axis=1)
    mn_xy = np.mean(omni_xy, axis=1)
    # Double check vecotorize code
    # for f_loop in range(n_freq):
    #     coh[f_loop]=(np.abs(mn_xy[f_loop]) ** 2) / (mn_xx[f_loop]*mn_yy[f_loop])
    coh = np.real(np.divide(np.abs(mn_xy) ** 2, np.multiply(xx, mn_yy)))
    return coh


def omni_cohgram(data, seed_chan_id, Sf, wind_len, wind_step, time_sec):
    """ Uses omni_coh to create a coherence gram between a seed channel and all other channels
    Inputs:
     data - chan x time matrix
     seed_chan_id - index of target channel
     Sf - sampling rate
     wind_len - # of time points in moving window
     wind_step - # of time points to advance moving window
     time_sec - the time (in seconds) of each time point in data"""

    n_tpt = data.shape[1]
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))

    # Compute # of time windows
    n_wind = len(wind_cntr)

    # Compute frequencies
    nyq = Sf / 2
    T = wind_len / Sf  # length of moving window in seconds
    freq_spacing = 1 / T
    f = np.linspace(0, nyq, 1 + nyq / freq_spacing)

    #Preallocate memory
    n_freq = len(f)  # of frequencies
    cohgram = np.zeros((n_freq, n_wind))
    cohgram_sec = np.zeros(n_wind)

    cursor = 0
    for wind in range(n_wind):
        cohgram[:,wind]=omni_coh(data[:,cursor:cursor + wind_len], seed_chan_id, wind_len, n_freq)
        cohgram_sec[wind] = np.mean(time_sec[cursor:cursor + wind_len])
        cursor += wind_step

    return cohgram, f, cohgram_sec

def bp_coh_omni(data, seed_chan_id, srate, wind_len, wind_step, tpts_sec, bands):
    """ Note that coherence values are arcsin transformed before averaging"""
    n_band=len(bands)
    use_ids=dict()
    # ToDo: ignore DC and harmonics
    cohgram, f, cohgram_sec =omni_cohgram(data, seed_chan_id, srate, wind_len, wind_step, tpts_sec)
    # cohgram, f, cohgram_sec=mt_cohgram(data1, data2, srate, wind_len, wind_step, n_tapers, tpts_sec)
    cohgram=dg.asin_trans(cohgram)
    n_wind = len(cohgram_sec)
    bp_coh = np.zeros((n_band, n_wind))
    # figure out which freqs to use
    for b in bands:
        use_ids[b] = np.logical_and(f > b[0], f < b[1])
    for b_ct, b in enumerate(bands):
        bp_coh[b_ct, :] = np.mean(cohgram[use_ids[b], :], axis=0)
    return bp_coh, cohgram_sec


def bp_coh_mt(data1, data2, srate, wind_len, wind_step, n_tapers, tpts_sec, bands):
    """ Note that coherence values are arcsin transformed before averaging"""
    n_band=len(bands)
    use_ids=dict()
    # ToDo: ignore DC and harmonics
    cohgram, f, cohgram_sec=mt_cohgram(data1, data2, srate, wind_len, wind_step, n_tapers, tpts_sec)
    cohgram=dg.asin_trans(cohgram)
    n_wind = len(cohgram_sec)
    bp_coh = np.zeros((n_band, n_wind))
    # figure out which freqs to use
    for b in bands:
        use_ids[b] = np.logical_and(f > b[0], f < b[1])
    for b_ct, b in enumerate(bands):
        bp_coh[b_ct, :] = np.mean(cohgram[use_ids[b], :], axis=0)
    return bp_coh, cohgram_sec


def cmpt_plvgram(data1, data2, Sf, wind_len, wind_step, tpts_sec):
    """ Computes phase locking value between two channels at all frequencies via a
    hamming window DFT."""
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))
    n_tpt = len(data1)
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)

    # Compute # of time windows
    n_wind = len(wind_cntr)

    # Hamming taper
    tpr = np.hamming(wind_len)

    # Compute frequencies
    nyq = Sf / 2
    T = wind_len / Sf  # length of moving window in seconds
    freq_spacing = 1 / T
    f = np.linspace(0, nyq, 1 + nyq / freq_spacing)
    n_freq = len(f)

    # preallocate mem
    delta_phz = np.zeros((n_freq, n_wind))
    dft_sec = np.zeros(n_wind)

    # Loop over iEEG windows, do DFT, and calculate phase difference
    cursor = 0
    for wind in range(n_wind):
        dft_sec[wind] = np.mean(tpts_sec[cursor:cursor + wind_len])

        # norm='ortho' means that no scaling factor is necessary to do inverse fft
        spec1 = np.fft.rfft(np.multiply(data1[cursor:cursor + wind_len], tpr), wind_len, norm='ortho')
        phase1 = np.angle(spec1)

        spec2 = np.fft.rfft(np.multiply(data2[cursor:cursor + wind_len], tpr), wind_len, norm='ortho')
        phase2 = np.angle(spec2)

        delta_phz[:, wind] = phase1 - phase2
        cursor += wind_step

    # Loop over phase windows and calculate PLV
    p_Sf = 1 / (dft_sec[1] - dft_sec[0])  # DFT moving window sampling rate
    p_wind_len = int(np.round(p_Sf))  # one second window
    #p_wind_len = int(np.round(3*p_Sf))  # three second window
    p_wind_step = 1 #move one DFT time window at a time
    p_wind_cntr = np.arange(p_wind_len, n_wind, p_wind_step)
    # Compute # of time windows
    p_n_wind = len(p_wind_cntr)

    # Preallocate mem
    pgram = np.zeros((n_freq, p_n_wind))
    p_sec = np.zeros(p_n_wind)

    cursor = 0
    for wind in range(p_n_wind):
        pgram[:, wind] = np.abs(np.mean(np.exp(1j * delta_phz[:, cursor:cursor + p_wind_len]), axis=1))
        p_sec[wind] = np.mean(dft_sec[cursor:cursor + p_wind_len])
        cursor += p_wind_step

    return pgram, f, p_sec


def cmpt_bp_plv(data1, data2, srate, wind_len, wind_step, tpts_sec, bands):
    """ Computes phase locking value between two channels averaged within a list of
    frequency bands via a hamming window DFT. Note that PLV values are arcsin
    transformed before averaging"""
    n_band=len(bands)
    use_ids=dict()
    # ToDo: ignore DC and harmonics
    pgram, f, p_sec=cmpt_plvgram(data1,data2,srate,wind_len,wind_step,tpts_sec)
    pgram=dg.asin_trans(pgram)
    n_wind = len(p_sec)
    bp_plv = np.zeros((n_band, n_wind))
    # figure out which freqs to use
    for b in bands:
        use_ids[b] = np.logical_and(f > b[0], f < b[1])
    for b_ct, b in enumerate(bands):
        bp_plv[b_ct, :] = np.mean(pgram[use_ids[b], :], axis=0)
    return bp_plv, p_sec


def omni_plvgram(data, seed_chan_id, Sf, wind_len, wind_step, tpts_sec):
    """ Uses plvgram to create a 'phase locking value-gram' between a seed channel and all other channels
    Note that plv values are arcsin transformed before averaging
    Inputs:
     data - chan x time matrix
     seed_chan_id - index of seed channel
     Sf - sampling rate
     wind_len - # of time points in moving window
     wind_step - # of time points to advance moving window
     tpts_sec - the time (in seconds) of each time point in data"""

    # This could be made more efficient by computing the DFT for the seed channel once.
    # Right now it is recomputed for every comparison with a non-seed channel
    n_chan = data.shape[0]
    n_tpt = data.shape[1]
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))

    # Compute # of time windows
    n_wind = len(wind_cntr)

    # Compute frequencies
    nyq = Sf / 2
    T = wind_len / Sf  # length of moving window in seconds
    freq_spacing = 1 / T
    f = np.linspace(0, nyq, 1 + nyq / freq_spacing)

    #Preallocate memory
    n_freq = len(f)  # of frequencies
    first_chan=True
    for chan in range(n_chan):
        if chan!=seed_chan_id:
            temp_pgram, f, omni_pgram_sec=cmpt_plvgram(data[seed_chan_id,:],data[chan,:],
                                         Sf,wind_len,wind_step,tpts_sec)
            temp_pgram=dg.asin_trans(temp_pgram)
            if first_chan==True:
                omni_pgram=temp_pgram
                first_chan==False
            else:
                omni_pgram+=temp_pgram
    omni_pgram=omni_pgram/(n_chan-1) # convert sum to mean

    return omni_pgram, f, omni_pgram_sec


def mt_sgram(data, srate, wind_len, wind_step, n_tapers, time_sec):
    """ Computes a multitaper DFT spectrogram of a channel.
    Returns: sgram, f, sgram_sec"""
    n_tpt = len(data)
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))
    # Compute # of time windows
    n_wind = len(wind_cntr)

    cursor = 0
    for wind in range(n_wind):
        spec, f = mtspec.multitaper.mtspec(data[cursor:cursor + wind_len], 1 / srate, n_tapers)
        if cursor == 0:
            n_freq = len(f) # of frequencies
            sgram = np.zeros((n_freq, n_wind))
            sgram_sec = np.zeros(n_wind)
        sgram[:, wind] = 10 * np.log10(spec)
        sgram_sec[wind] = np.mean(time_sec[cursor:cursor + wind_len])
        cursor += wind_step
    return sgram, f, sgram_sec


def hamming_sgram(data, srate, wind_len, wind_step, time_sec):
    n_tpt = len(data)
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))
    # Compute # of time windows
    n_wind = len(wind_cntr)

    # Hamming taper
    tpr = np.hamming(wind_len)

    # Compute frequencies
    nyq=srate/2
    T=wind_len/srate # length of moving window in seconds
    freq_spacing = 1 / T
    f = np.linspace(0, nyq, 1 + nyq / freq_spacing)
    n_freq = len(f)

    # pre-allocate memory
    sgram = np.zeros((n_freq, n_wind))
    sgram_sec = np.zeros(n_wind)

    cursor = 0
    for wind in range(n_wind):
        spec=np.fft.rfft(np.multiply(data[cursor:cursor + wind_len],tpr), wind_len, norm='ortho')
        sgram[:, wind] = 10 * np.log10(np.abs(spec))
        sgram_sec[wind] = np.mean(time_sec[cursor:cursor + wind_len])
        cursor += wind_step
    return sgram, f, sgram_sec


def mt_cohgram(data1, data2, srate, wind_len, wind_step, n_tapers, time_sec):
    n_tpt = len(data1)
    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step))
    # Compute # of time windows
    n_wind = len(wind_cntr)

    # Compute # of frequencies
    cursor = 0
    for wind in range(n_wind):
        mt_coh = mtspec.multitaper.mt_coherence(1 / srate, data1[cursor:cursor + wind_len],
                                                data2[cursor:cursor + wind_len],
                                                int(n_tapers + 1) / 2, n_tapers, int(srate/2), 0.95,
                                                freq=True, cohe=True, phase=True)
        f = mt_coh['freq']
        if cursor == 0:
            n_freq = len(f)
            cohgram = np.zeros((n_freq, n_wind))
            cohgram_sec = np.zeros(n_wind)
        cohgram[:, wind] = mt_coh['cohe']
        cohgram_sec[wind] = np.mean(time_sec[cursor:cursor + wind_len])
        cursor += wind_step
    return cohgram, f, cohgram_sec


def z_norm(data, time_sec, onset_sec):
    """ converts chan x time matrix to z-scores via baseline period"""
    # Note, I don't need to return anything because the variable is passed by reference
    n_chan, n_tpt = data.shape
    use_tpts = time_sec < onset_sec
    mn = np.mean(data[:, use_tpts], axis=1)
    sd = np.std(data[:, use_tpts], axis=1)
    for chan in range(n_chan):
        data[chan, :] = (data[chan, :] - mn[chan]) / sd[chan]


def ptile_norm(data, time_sec, onset_sec):
    """ converts chan x time matrix to percentiles via baseline period"""
    n_chan, n_tpt = data.shape
    use_tpts = time_sec < onset_sec
    n_use_tpts=len(use_tpts)
    ptiles = np.zeros((n_chan, n_use_tpts))
    for chan in range(n_chan):
        ptiles[chan, :] = [stats.percentileofscore(data[chan,:n_use_tpts],
                                                          a, 'rank') for a in data[chan,:n_use_tpts]]
    return ptiles, time_sec[:n_use_tpts]


def import_ieeg(ieeg_mat_fname):
    """ Import iEEG data from mat file and remove mean of each channel
    For example:
    ieeg, Sf, tpts_sec=import_ieeg('NA_d1_sz2.mat')"""
    path_dict = get_path_dict()
    ieeg_root=path_dict['ieeg_root']
    # ieeg_root=get_ieeg_root_dir()
    sub=ieeg_mat_fname.split('_')[0]
    #ieeg_dir=os.path.join(ieeg_root,sub,'Data')
    ieeg_dir=os.path.join(ieeg_root,sub,'EEG_MAT')
    ieeg_fname=os.path.join(ieeg_dir,ieeg_mat_fname)
    print('Loading %s' % ieeg_fname)
    mat=sio.loadmat(ieeg_fname)
    Sf=mat['Sf'][0][0]
    ieeg=mat['matrix_bi']
    ieeg=ieeg.T
    n_nan=np.sum(np.isnan(ieeg))
    if n_nan>0:
        print('Warning: Setting %d NaN values in the data to 0.' % n_nan)
        ieeg=np.nan_to_num(ieeg)
    n_chan, n_tpt=ieeg.shape
    # detrend each channel (first channel of some files has crazy trend)
    ieeg=signal.detrend(ieeg,axis=1)
    # remove mean of each channel
    # for chan_loop in range(n_chan):
    #     ieeg[chan_loop,:]=ieeg[chan_loop,:]-np.mean(ieeg[chan_loop,:])
    tpts_sec=np.arange(0,n_tpt)/Sf
    tpts_sec=tpts_sec.T
    return ieeg, Sf, tpts_sec


def import_chan_labels(sub):
    """ Import channel labels as list of strings. For example:
    chan_labels=import_chan_labels('NA')"""
    path_dict = get_path_dict()
    ieeg_root=path_dict['ieeg_root']
    # ieeg_root=get_ieeg_root_dir()
    chan_fname=os.path.join(ieeg_root,sub,
                        sub+'_channel_info.csv')
    print('Loading %s' % chan_fname)
    chan_labels_df=pd.read_csv(chan_fname,names=['label'])
    # Convert to list and remove redundant electrode stem for second electrode in each bipolar pair
    chan_labels=clean_labels(list(chan_labels_df['label']))
    return chan_labels


def clin_onset_tpt_and_chan(szr_name, onset_df):
    import re
    name_splt=szr_name.split('_')
    day=int(name_splt[1][1:])
    szr=int(re.findall('[0-9]+',name_splt[2])[0])
    print('Getting clinical onset time for %s: Day %d, Szr %d' % (name_splt[0],day,szr))

    # temp_df=onset_df[onset_df['DAY']==day and onset_df['SZR#']==2]
    day_ids=onset_df[onset_df['DAY']==day].index
    szr_ids=onset_df[onset_df['SZR#']==szr].index
    use_id=day_ids.intersection(szr_ids)
    if len(use_id)==0:
        onset_tpt=NaN
        onset_chan=NaN
    else:
        onset_tpt=onset_df['ONSET_TPT'].iloc[use_id].values[0]
        onset_chan=onset_df['DG_ONSET_CHAN'].iloc[use_id].values[0]
    return int(onset_tpt), onset_chan


def cmpt_postonset_stim_latency(class_hat,class_true,Fs):
    """ Computes the latency of the earliest 'stimulation' after clinician onset
    all values are positive in units of seconds or None
    None means that no stimulation would have occurred after clinician onset

    Inputs:
     class_hat - binary array of predicted ictal class (1=ictal/stimuate)
     class_true - array of clinician ictal class (1=ictal, 0=preictal, -1=late ictal)

    Outputs:
     onset_dif_sec - the time in seconds of the first "stimulation" following clinician onset (None
       if no post-onset stimulation triggered)
     preonset_stim - 1 if there was a pre-onset stimulation triggered, 0 otherwise

    """
    pos_ids = np.where(np.multiply(class_hat == 1, class_true != 0))
    ictal_ids = np.where(class_true == 1)
    if len(pos_ids[0]) > 0:
        onset_dif_sec = (pos_ids[0][0] - ictal_ids[0][0]) * Fs
    else:
        onset_dif_sec=None

    pre_ids = np.where(np.multiply(class_hat == 1, class_true == 0))
    if len(pre_ids[0]) > 0:
        preonset_stim = 1
    else:
        preonset_stim = 0
    return onset_dif_sec, preonset_stim


def cmpt_stim_latency(class_hat,class_true,Fs):
    """ Computes the latency of the earliest 'stimulation' relative to clinician onset
    + value means that stimulation is AFTER onset
    - value means that stimulation is BEFORE onset
    None means that no stimulation would have occurred

    Inputs:
     class_hat - binary array of predicted ictal class (1=ictal/stimuate)
     class_true - array of clinician ictal class (1=ictal, 0=preictal, -1=late ictal)
    """
    pos_ids = np.where(class_hat == 1)
    ictal_ids = np.where(class_true == 1)
    if len(pos_ids[0]) > 0:
        onset_dif_sec = (pos_ids[0][0] - ictal_ids[0][0]) * Fs
    else:
        onset_dif_sec=None
    return onset_dif_sec


def cmpt_vltg_ftrs(data, wind_len, wind_step, tpts_sec):
    """ Returns the following voltage domain features computed with a moving window
     Input:
       data - ??needs to be a 1D numpy matrix (not a 1D array)"""
    n_chan, n_tpt = data.shape
    print('chans %d' % n_chan)
    print('tpts %d' % n_tpt)

    wind_cntr = np.arange(wind_len, n_tpt, wind_step)
    wind_len = int(np.round(wind_len))
    wind_step = int(np.round(wind_step)) # Compute # of time windows

    n_wind = len(wind_cntr)
    ftr_list=['rms','std','kurtosis','skew','line_len']
    n_ftr=len(ftr_list)

    # preallocate mem
    vltg_ftrs = np.zeros((n_chan, n_ftr, n_wind))
    moving_wind_sec=np.zeros(n_wind)

    # TODO: apply a taper before computing features?
    #  tpr = np.hamming(wind_len)

    # Compute central time point (in seconds) of moving window
    cursor = 0
    for wind in range(n_wind):
        moving_wind_sec[wind]= np.mean(tpts_sec[cursor:cursor + wind_len])
        cursor += wind_step

    # Compute features at each channel via a moving window
    for chan in range(n_chan):
        cursor = 0
        for wind in range(n_wind):
            # TODO: apply a taper before computing features?
            #np.multiply(data[cursor:cursor + wind_len], tpr),wind_len, norm='ortho')

            for ftr_loop in range(n_ftr):
                dat=np.squeeze(data[chan,cursor:cursor + wind_len])
                if ftr_list[ftr_loop]=='rms':
                    vltg_stat=np.log(1+np.sqrt(np.mean(dat**2))) # log transform to make data more Gaussian
                elif ftr_list[ftr_loop]=='std':
                    vltg_stat=np.log(1+np.std(dat)) # log transform to make data more Gaussian
                elif ftr_list[ftr_loop]=='kurtosis':
                    vltg_stat=np.log(1+scipy.stats.kurtosis(dat,axis=0,fisher=False,bias=True))
                elif ftr_list[ftr_loop]=='skew':
                    vltg_stat = scipy.stats.skew(dat, axis=0, bias=True)
                elif ftr_list[ftr_loop]=='line_len':
                    if cursor>0:
                        dat_delt=dat-data[chan,(cursor-1):(cursor-1 + wind_len)]
                        # true line length
                        #vltg_stat = np.mean(np.sqrt(dat_delt**2+np.ones(wind_len)*(1/srate)**2))
                        # approximate line length
                        vltg_stat = np.log(1+np.mean(np.abs(dat_delt))) # log transform to make data more Gaussian
                        #vltg_stat = np.mean(np.abs(dat_delt))
                    else:
                        vltg_stat=0
                else:
                    raise ValueError('Unknown feature specified')
                vltg_ftrs[chan, ftr_loop, wind]=vltg_stat
            # sgram_sec[wind] = np.mean(time_sec[cursor:cursor + wind_len])
            cursor +=wind_step

    return vltg_ftrs, moving_wind_sec, ftr_list


def perf_msrs(true_class, hat_class):
    """ bal_acc, sens, spec=perf_msrs(true_class, hat_class)
    Computes balanced accuracy, sensitivity, and specificity of classifier
    true_class and hat_class need to be binary"""
    jive = (hat_class == true_class)
    sens = np.sum(jive[true_class == 1]) / np.sum(true_class == 1)
    spec = np.sum(jive[true_class == 0]) / np.sum(true_class == 0)
    bal_acc = (sens + spec) / 2
    return bal_acc, sens, spec


def apply_model_2_file_list(model, ftr_fname_list, ftr_names, ftr_nrm_dicts, sub, n_ftr_dim, ext_list, edge_pts, szr_ant):
    """ Applies a classifier to a list of EU feature files
    Inputs:
    model - sklearn model
    ftr_fname - list of EU feature file stems
    ftr_nrm_dicts - list of dictionaries of the mean and SD for each feature
    sub - patient ID # (e.g., 1096)
    n_ftr_dim - Total number of features
    ext_list - list of the feature file extensions
    edge_pts - # of time windows at start of file to ignore due to possible edge effects. Should be 1177 for 10 Hz
               moving window srate
    szr_ant - If true, seizure anticipation class labels are used, which target a few seconds before and after onset
              If false, all time points between onset and offset are considered targets

    NOTE! ext_list, ftr_fname_list, and ftr_nrm_dicts all need to be in the same order

    Outputs:
      valid_bal_acc, valid_sens, valid_spec, valid_acc:
      (self-explanatory)"""

    dir_dict = get_path_dict()
    ftrs_root = dir_dict['ftrs_root']

    n_total_hit = 0
    n_total_wind = 0
    n_valid_szr_wind = 0
    n_hit_valid_szr_wind = 0
    n_valid_non_szr_wind = 0
    n_hit_valid_non_szr_wind = 0
    n_files=len(ftr_fname_list)
    if szr_ant:
        class_path = os.path.join(ftrs_root, 'EU_SZR_ANT_CLASS', sub)
    else:
        class_path = os.path.join(ftrs_root, 'EU_SZR_CLASS', sub)
    print('Ftrs being used are {}'.format(ftr_names))
    for fname_ct, ftr_fname in enumerate(ftr_fname_list):
        if (fname_ct % 10) == 0:
            print('Loading file %d of %d' % (fname_ct + 1, n_files))
        # Load the class labels for this time period
        full_fname = os.path.join(class_path, ftr_fname + '_szr_class.npz')
        class_dict = np.load(full_fname)
        temp_class = class_dict['szr_class'][edge_pts:] #ignore time points susceptible to edge effects
        use_temp_class_bool=temp_class>=0 # Ignore szr time points 9 sec or later after onset
        n_trim_wind = len(temp_class)
        # n_wind = len(temp_class)
        # n_trim_wind=n_wind-edge_pts

        # Preallocate memory
        temp_data = np.zeros((n_ftr_dim, n_trim_wind))

        # Load all the various features for this time period
        ftr_dim_ct = 0
        for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
            ftr_path = os.path.join(ftrs_root, temp_ftr_type, sub)
            full_fname = os.path.join(ftr_path, ftr_fname + ext_list[ftr_type_ct])
            ftr_dict = np.load(full_fname)
            temp_ftr_dim, temp_n_wind = ftr_dict['ftrs'].shape
            temp_data[ftr_dim_ct:ftr_dim_ct + temp_ftr_dim, :] = ftr_dict['ftrs'][:,edge_pts:]
            # Normalize ftrs
            for ftr_loop in range(temp_ftr_dim):
                temp_data[ftr_dim_ct + ftr_loop, :] = (temp_data[ftr_dim_ct + ftr_loop, :] -
                                                       ftr_nrm_dicts[ftr_type_ct]['nrm_mn'][ftr_loop])/ftr_nrm_dicts[ftr_type_ct]['nrm_sd'][ftr_loop]
            ftr_dim_ct += temp_ftr_dim

        # Apply classifier
        #temp_class_hat = model.predict(temp_data.T)  # outputs 0 or 1
        temp_class=temp_class[use_temp_class_bool]
        temp_class_hat = model.predict(temp_data[:,use_temp_class_bool].T)  # outputs 0 or 1
        # Tally accuracy
        jive = (temp_class_hat == temp_class)
        n_valid_szr_wind += np.sum(temp_class)
        n_hit_valid_szr_wind += np.sum(jive[temp_class == 1])
        n_valid_non_szr_wind += np.sum(temp_class == 0)
        n_hit_valid_non_szr_wind += np.sum(jive[temp_class == 0])
        n_total_hit += np.sum(jive)
        n_total_wind += n_trim_wind

    valid_sens = n_hit_valid_szr_wind / n_valid_szr_wind
    valid_spec = n_hit_valid_non_szr_wind / n_valid_non_szr_wind
    valid_bal_acc = (valid_sens + valid_spec) / 2
    valid_acc = n_total_hit / n_total_wind

    return valid_bal_acc, valid_sens, valid_spec, valid_acc
