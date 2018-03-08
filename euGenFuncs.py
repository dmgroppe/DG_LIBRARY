import os
import scipy.io as sio
import numpy as np
import ieeg_funcs as ief
import dgFuncs as dg

# Function for extracting channel names from filename
def chan_labels_from_fname(in_file):
    """ Extracts the bipolar channel label from a feature file name """
    just_fname=in_file.split('/')[-1]
    jf_splt=just_fname.split('_')
    chan_label=jf_splt[1]+'-'+jf_splt[2]
    return chan_label

def data_size_and_fnames(sub_list, ftr_root, ftr, dsamp_pcnt=1):
    """ Get size of data (and filenames) """
    grand_non_fnames = list()
    grand_szr_fnames = list()
    grand_n_szr_wind = 0
    grand_n_non_wind = 0
    non_file_subs=list()
    szr_file_subs = list()
    non_file_chans=list()
    szr_file_chans = list()
    # TODO need to record list of subjects and channels to make sure they are the same across features
    ftr_path=os.path.join(ftr_root,ftr)
    for sub in sub_list:
        print('Working on sub %d' % sub)
        non_fnames = list()
        szr_fnames = list()
        subsamp_fnames = list()

        # Get filenames (and full path)
        sub_ftr_path = os.path.join(ftr_path, str(sub))
        for f in os.listdir(sub_ftr_path):
            if f.endswith('non.mat'):
                non_fnames.append(os.path.join(sub_ftr_path, f))
                non_file_subs.append(sub)
                non_file_chans.append(chan_labels_from_fname(f))
            elif f.endswith('subsamp.mat'):
                subsamp_fnames.append(os.path.join(sub_ftr_path, f)) # This isn't actually needed for anythin
            elif f.endswith('.mat') and f.startswith(str(sub) + '_'):
                szr_fnames.append(os.path.join(sub_ftr_path, f))
                szr_file_subs.append(sub)
                szr_file_chans.append(chan_labels_from_fname(f))

        print('%d non-szr files found' % len(non_fnames))
        print('%d szr files found' % len(szr_fnames))

        if ftr=='PLV_SE':
            n_ftrs=2
        elif ftr=='SE':
            n_ftrs=1
        else:
            print('Unrecognized feature %s' % ftr)
            exit()

        # Loop over NON-szr files to get total # of windows
        n_non_wind = 0
        ftr_dim = 0
        for f in non_fnames:
            #             in_file=os.path.join(ftr_path,f)
            #             temp_ftrs=sio.loadmat(in_file)
            temp_ftrs = sio.loadmat(f)
            n_non_wind += int(np.round(dsamp_pcnt*temp_ftrs['nonszr_se_ftrs'].shape[1]))
            if ftr_dim == 0:
                ftr_dim = temp_ftrs['nonszr_se_ftrs'].shape[0]*n_ftrs
            elif ftr_dim != temp_ftrs['nonszr_se_ftrs'].shape[0]*n_ftrs:
                raise ValueError('# of features in file does match previous files')

        print('%d total # of NON-szr time windows for this sub' % n_non_wind)

        # Loop over SZR files to get total # of windows
        n_szr_wind = 0
        for f in szr_fnames:
            #             in_file=os.path.join(ftr_path,f)
            #             temp_ftrs=sio.loadmat(in_file)
            temp_ftrs = sio.loadmat(f)
            n_szr_wind += int(np.round(dsamp_pcnt*temp_ftrs['se_ftrs'].shape[1]))
        print('%d total # of SZR time windows for this sub' % n_szr_wind)

        grand_non_fnames += non_fnames
        grand_szr_fnames += szr_fnames
        grand_n_szr_wind += n_szr_wind
        grand_n_non_wind += n_non_wind

    ftr_info_dict=dict()
    ftr_info_dict['szr_file_chans']=szr_file_chans
    ftr_info_dict['non_file_chans'] = non_file_chans
    ftr_info_dict['szr_file_subs'] = szr_file_subs
    ftr_info_dict['non_file_subs'] = non_file_subs
    ftr_info_dict['ftr_dim'] = ftr_dim
    ftr_info_dict['grand_n_non_wind']=grand_n_non_wind
    ftr_info_dict['grand_n_szr_wind']=grand_n_szr_wind
    ftr_info_dict['grand_non_fnames']=grand_non_fnames
    ftr_info_dict['grand_szr_fnames']=grand_szr_fnames

    return ftr_info_dict


def lim_ftr_range(raw_ftrs):
    # Load normalization parameters that will further rescale and recenter data
    # to get informative range from -3.99 to 3.99
    path_dict = ief.get_path_dict()
    nrm_fname=os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','KDOWNSAMP','norm_factors.npz')
    #npz_nrm=np.load('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/norm_factors.npz')
    npz_nrm = np.load(nrm_fname)
    upper_bnd=3.99
    lower_bnd=-3.99
    for ftr_ct in range(raw_ftrs.shape[0]):
        # TODO make sure features in npz_nrm match ftrs of current data
        #raw_ftrs[ftr_ct,:]=(raw_ftrs[ftr_ct,:]-npz_nrm['cntr'][ftr_ct])/npz_nrm['div_fact'][ftr_ct]
        raw_ftrs[ftr_ct, :] = raw_ftrs[ftr_ct, :]/npz_nrm['div_fact'][ftr_ct]
        raw_ftrs[ftr_ct, :] = raw_ftrs[ftr_ct, :] + npz_nrm['cntr'][ftr_ct]

        raw_ftrs[ftr_ct,raw_ftrs[ftr_ct,:]>upper_bnd]=upper_bnd # set max possible value
        raw_ftrs[ftr_ct, raw_ftrs[ftr_ct, :] < lower_bnd] = lower_bnd  # set min possible value
        #print('Min/Max ftr %f %f' % (np.min(raw_ftrs[ftr_ct, :]),np.max(raw_ftrs[ftr_ct, :])))


def import_data(szr_fnames, non_fnames, szr_subs, non_subs, n_szr_wind, n_non_wind, ftr_dim, dsamp_pcnt=1, bnded=True):
    # ftr_path=os.path.join(ftr_root,str(sub))

    # Preallocate memory
    ftrs = np.zeros((ftr_dim, n_szr_wind + n_non_wind))
    targ_labels = np.zeros(n_szr_wind + n_non_wind)
    sub_ids=np.zeros(n_szr_wind + n_non_wind)

    if ftr_dim==60:
        using_plv=True

    # Import non-szr data
    ptr = 0
    mns_dict = dict()
    sds_dict = dict()
    chan_list=list()
    for f_ct, f in enumerate(non_fnames):
        chan_label = str(non_subs[f_ct])+'_'+chan_labels_from_fname(f)
        print(chan_label)
        chan_list.append(chan_label)

        if using_plv==False:
            # Load subsampled data (possibly contains both szr and non-szr data)
            subsamp_f=f[:-7]+'subsamp.mat'
            temp_ftrs = sio.loadmat(subsamp_f)
            raw_ftrs = temp_ftrs['subsamp_se_ftrs']
            # Z-score features USE THE CODE BELOW
            # ORIG NORMALIZATION: Remove 50% most extreme points and then z-score
            # temp_mns, temp_sds = dg.trimmed_normalize(raw_ftrs, 0.25, zero_nans=False, verbose=False) #normalization is done in place
            # mns_dict[chan_label] = temp_mns
            # sds_dict[chan_label] = temp_sds
        else:
            # For SE+PLV features I didn't have time to randomly subsample the data (both ictal and nonictal),
            # so I normalize by nonszr samples
            temp_ftrs = sio.loadmat(f)
            raw_ftrs = np.concatenate((temp_ftrs['nonszr_se_ftrs'], temp_ftrs['nonszr_plv_ftrs']))


        # This subtracts the median from each time series and divides by the IQR
        print('Subtracting median and dividing by IQR')
        temp_mns, temp_sds = dg.median_normalize(raw_ftrs, zero_nans=False, verbose=False)  # normalization is done in place
        mns_dict[chan_label] = temp_mns
        sds_dict[chan_label] = temp_sds
        # raw ftrs is ftr x time window


        # Load nonszr data
        print('Loading file %s' % f)
        temp_ftrs = sio.loadmat(f)
        temp_n_wind = int(np.round(temp_ftrs['nonszr_se_ftrs'].shape[1]*dsamp_pcnt))
        temp_wind_ids=np.random.permutation(temp_ftrs['nonszr_se_ftrs'].shape[1])[:temp_n_wind]
        if ftr_dim==60:
            raw_ftrs =np.concatenate((temp_ftrs['nonszr_se_ftrs'][:, temp_wind_ids], temp_ftrs['nonszr_plv_ftrs'][:, temp_wind_ids]))
        else:
            raw_ftrs = temp_ftrs['nonszr_se_ftrs'][:,temp_wind_ids]
        # Z-score based on trimmed subsampled means, SDs
        dg.applyNormalize(raw_ftrs, mns_dict[chan_label], sds_dict[chan_label])
        #if bnded: ?? TODO use lim_ftr_range(raw_ftrs) # re-normalize and truncate to -3.99 to 3.99
        ftrs[:, ptr:ptr + temp_n_wind] = raw_ftrs
        targ_labels[ptr:ptr + temp_n_wind] = 0
        sub_ids[ptr:ptr + temp_n_wind] = non_subs[f_ct]
        ptr += temp_n_wind

    # Import szr data
    for f_ct, f in enumerate(szr_fnames):
        #chan_label = chan_labels_from_fname(f)
        chan_label = str(szr_subs[f_ct]) + '_' + chan_labels_from_fname(f)

        temp_ftrs = sio.loadmat(f)
        temp_n_wind = int(np.round(temp_ftrs['se_ftrs'].shape[1]*dsamp_pcnt))
        temp_wind_ids = np.random.permutation(temp_ftrs['se_ftrs'].shape[1])[:temp_n_wind]
        if ftr_dim == 60:
            raw_ftrs = np.concatenate((temp_ftrs['se_ftrs'][:,temp_wind_ids],temp_ftrs['plv_ftrs'][:,temp_wind_ids]))
        else:
            raw_ftrs = temp_ftrs['se_ftrs'][:, temp_wind_ids]
        # Z-score based on trimmed subsampled means, SDs
        dg.applyNormalize(raw_ftrs, mns_dict[chan_label], sds_dict[chan_label])
        #if bnded TODO use ?? lim_ftr_range(raw_ftrs) # re-normalize and truncate to -3.99 to 3.99

        ftrs[:, ptr:ptr + temp_n_wind] = raw_ftrs
        targ_labels[ptr:ptr + temp_n_wind] = 1
        sub_ids[ptr:ptr + temp_n_wind] = szr_subs[f_ct]
        ptr += temp_n_wind

    # Load normalization parameters that will further rescale and recenter data
    # to get informative range from -3.99 to 3.99
    # path_dict = ief.get_path_dict()
    # nrm_fname=os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','KDOWNSAMP','norm_factors.npz')
    # #npz_nrm=np.load('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/norm_factors.npz')
    # npz_nrm = np.load(nrm_fname)
    # upper_bnd=3.99
    # lower_bnd=-3.99
    # for ftr_ct in range(raw_ftrs.shape[0]):
    #     # TODO make sure features in npz_nrm match ftrs of current data
    #     #raw_ftrs[ftr_ct,:]=(raw_ftrs[ftr_ct,:]-npz_nrm['cntr'][ftr_ct])/npz_nrm['div_fact'][ftr_ct]
    #     raw_ftrs[ftr_ct, :] = raw_ftrs[ftr_ct, :]/ npz_nrm['div_fact'][ftr_ct]
    #     raw_ftrs[ftr_ct, :] = raw_ftrs[ftr_ct, :] - npz_nrm['cntr'][ftr_ct]
    #
    #     raw_ftrs[ftr_ct,raw_ftrs[ftr_ct,:]>upper_bnd]=upper_bnd # set max possible value
    #     raw_ftrs[ftr_ct, raw_ftrs[ftr_ct, :] < lower_bnd] = lower_bnd  # set min possible value
    #     print('Min/Max ftr %f %f' % (np.min(raw_ftrs[ftr_ct, :]),np.max(raw_ftrs[ftr_ct, :])))

    return ftrs.T, targ_labels, sub_ids