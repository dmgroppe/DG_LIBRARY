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

def data_size_and_fnames(sub_list, ftr_root, ftr):
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

        # Loop over NON-szr files to get total # of windows
        n_non_wind = 0
        ftr_dim = 0
        for f in non_fnames:
            #             in_file=os.path.join(ftr_path,f)
            #             temp_ftrs=sio.loadmat(in_file)
            temp_ftrs = sio.loadmat(f)
            n_non_wind += temp_ftrs['nonszr_se_ftrs'].shape[1]
            if ftr_dim == 0:
                ftr_dim = temp_ftrs['nonszr_se_ftrs'].shape[0]
            elif ftr_dim != temp_ftrs['nonszr_se_ftrs'].shape[0]:
                raise ValueError('# of features in file does match previous files')

        print('%d total # of NON-szr time windows for this sub' % n_non_wind)

        # Loop over SZR files to get total # of windows
        n_szr_wind = 0
        for f in szr_fnames:
            #             in_file=os.path.join(ftr_path,f)
            #             temp_ftrs=sio.loadmat(in_file)
            temp_ftrs = sio.loadmat(f)
            n_szr_wind += temp_ftrs['se_ftrs'].shape[1]
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

def import_data(szr_fnames, non_fnames, szr_subs, non_subs, n_szr_wind, n_non_wind, ftr_dim):
    # ftr_path=os.path.join(ftr_root,str(sub))

    # Preallocate memory
    ftrs = np.zeros((ftr_dim, n_szr_wind + n_non_wind))
    targ_labels = np.zeros(n_szr_wind + n_non_wind)
    sub_ids=np.zeros(n_szr_wind + n_non_wind)

    # Import non-szr data
    ptr = 0
    mns_dict = dict()
    sds_dict = dict()
    chan_list=list()
    for f_ct, f in enumerate(non_fnames):
        chan_label = str(non_subs[f_ct])+'_'+chan_labels_from_fname(f)
        print(chan_label)
        chan_list.append(chan_label)

        # Load subsampled data (possibly contains both szr and non-szr data)
        subsamp_f=f[:-7]+'subsamp.mat'
        temp_ftrs = sio.loadmat(subsamp_f)
        raw_ftrs = temp_ftrs['subsamp_se_ftrs']
        # Z-score features USE THE CODE BELOW
        temp_mns, temp_sds = dg.trimmed_normalize(raw_ftrs, 0.25, zero_nans=False, verbose=False) #normalization is done in place
        mns_dict[chan_label] = temp_mns
        sds_dict[chan_label] = temp_sds

        # Load nonszr data
        print('Loading file %s' % f)
        temp_ftrs = sio.loadmat(f)
        temp_n_wind = temp_ftrs['nonszr_se_ftrs'].shape[1]
        raw_ftrs = temp_ftrs['nonszr_se_ftrs']
        # Z-score based on trimmed subsampled means, SDs
        dg.applyNormalize(raw_ftrs, mns_dict[chan_label], sds_dict[chan_label])
        ftrs[:, ptr:ptr + temp_n_wind] = raw_ftrs
        targ_labels[ptr:ptr + temp_n_wind] = 0
        sub_ids[ptr:ptr + temp_n_wind] = non_subs[f_ct]
        ptr += temp_n_wind

    # Import szr data
    for f_ct, f in enumerate(szr_fnames):
        #chan_label = chan_labels_from_fname(f)
        chan_label = str(szr_subs[f_ct]) + '_' + chan_labels_from_fname(f)

        temp_ftrs = sio.loadmat(f)
        temp_n_wind = temp_ftrs['se_ftrs'].shape[1]
        raw_ftrs = temp_ftrs['se_ftrs']
        # Z-score based on trimmed subsampled means, SDs
        dg.applyNormalize(raw_ftrs, mns_dict[chan_label], sds_dict[chan_label])

        ftrs[:, ptr:ptr + temp_n_wind] = raw_ftrs
        targ_labels[ptr:ptr + temp_n_wind] = 1
        sub_ids[ptr:ptr + temp_n_wind] = szr_subs[f_ct]
        ptr += temp_n_wind

    return ftrs.T, targ_labels, sub_ids