import inifile
import numpy as np
import pdb,traceback,sys
import time
import os
import re
from time import mktime
from datetime import datetime

######## USEFUL FUNCS ########
def get_sample_times(layFileName):
    data, sections, subsections = inifile.inifile(layFileName, 'readall')  # sections and subsections currently unused
    for row in data:
        for entry in row:
            if entry == []:
                entry = ''

    # Read "sampletimes" section of lay file
    sample_times = []  # list of dictionaries
    for row in data:
        if row[0] == 'sampletimes':
            # print("sample: {}, time: {}".format(float(row[2]),float(row[3])))
            sample_times.append({'sample': float(row[2]), 'time': float(row[3])})
    return sample_times


def get_srate(layFileName):
    data, sections, subsections = inifile.inifile(layFileName, 'readall')  # sections and subsections currently unused
    for row in data:
        for entry in row:
            if entry == []:
                entry = ''
    # find fileinfo section of .lay file
    fileInfoArray = []
    for row in data:
        if row[0] == 'fileinfo':
            fileInfoArray.append(row)
    fileinfo = {}  # dictionary
    for row in fileInfoArray:
        fileinfo[row[2]] = row[3]
    return int(fileinfo['samplingrate'])


def tidy_chan_names(channel_names):
    tidy_names=list()
    for chan in channel_names:
        tidy_names.append(chan.split('-')[0])
    return tidy_names


def get_eeg_chan_names(channel_names):
    (keep_chan_ids, c_chan_ids)=find_c_chans(channel_names,verbose=False)
    eeg_chan_names=list()
    non_eeg_chans=['event','osat-ref','pr-ref']
    for id in keep_chan_ids:
        if channel_names[id] not in non_eeg_chans:
            eeg_chan_names.append(channel_names[id])
    return tidy_chan_names(eeg_chan_names)

def find_c_chans(channel_names,verbose=True):
    keep_chan_ids=list()
    c_chan_ids=list()
    for ct, chan in enumerate(channel_names):
        chans=chan.split('-')
        match=re.search(r"^c[0-9]+",chans[0])
        if match:
            #print('found '+match.group()) ## 'found word:cat'
            c_chan_ids.append(ct)
        else:
            keep_chan_ids.append(ct)
    if verbose==True:
        print('Removing %d c* chans as they are likely unused' % len(c_chan_ids))
    return keep_chan_ids, c_chan_ids

def rm_c_chans(ieeg,chan_names,verbose=True):
    if verbose==True:
        print('Total # of channels: %d' % ieeg.shape[0])
    keep_chan_ids, c_chan_ids=find_c_chans(chan_names,verbose)
    ieeg=ieeg[keep_chan_ids,:]
    pruned_chan_names=[chan_names[i] for i in keep_chan_ids]
    return ieeg, pruned_chan_names

def rm_event_chan(ieeg,chan_names,verbose=True):
    ev_id=None
    for ct, chan in enumerate(chan_names):
        if chan=='event':
            ev_id=ct
            break
    if ev_id==None:
        if verbose==True:
            print('No "event" channel found.')
    else:
        if verbose==True:
            print('Unique event channel values: {}'.format(np.unique(ieeg[ev_id,:])))
            print('Removing event channel')
        n_chan=ieeg.shape[0]
        keep_chans=np.setdiff1d(np.arange(0,n_chan),ev_id)
        ieeg=ieeg[keep_chans,:]
        orig_chan_names=chan_names
        chan_names=[orig_chan_names[i] for i in keep_chans]
    return ieeg, chan_names

def rm_noneeg_chan(ieeg,chan_names,verbose=True):
    noneeg_ids=[]
    for ct, chan in enumerate(chan_names):
        if chan in ['osat-ref','pr-ref']:
            noneeg_ids.append(ct)

    if len(noneeg_ids)>0:
        if verbose==True:
            print('Removing these non-eeg channels:')
            for a in noneeg_ids:
                print(chan_names[a])
        n_chan=ieeg.shape[0]
        keep_chans=np.setdiff1d(np.arange(0,n_chan),np.asarray(noneeg_ids,dtype=int))
        ieeg=ieeg[keep_chans,:]
        orig_chan_names=chan_names
        chan_names=[orig_chan_names[i] for i in keep_chans]
    return ieeg, chan_names


def avg_ref(ieeg,report=True):
    if report:
        print('Taking mean time series across all channels and subtracting it from each channel.')
    mn=np.mean(ieeg,axis=0)
    n_chan=ieeg.shape[0]
    for c in range(n_chan):
        ieeg[c,:]=ieeg[c,:]-mn
    return ieeg, mn


def starttime_anon(starttime_str):
    """Returns the start time minus the year (in order to anonymize it.
    starttime_str should have a format like this '12-Jun-2001 08:44:52'
    """
    temp=starttime_str.split(' ')
    return temp[0][:-5]+' '+temp[1]


def sample_times_sec(sample_time_list,n_tpt,tstep_sec):
    """ Returns a vector with the time of day (in seconds) corresponding to each time point of EEG data"""
    # collect samples with time stamps
    clocked_samples = list()
    clocked_sample_times_sec = list()
    for stime in sample_time_list:
        clocked_samples.append(int(stime['sample']))
        clocked_sample_times_sec.append(stime['time'])

    time_of_day_sec=np.zeros(n_tpt)
    for t in range(n_tpt):
        if t in clocked_samples: # this is the subset of samples with clock times, which is used for clock synchronization
            t_id=clocked_samples.index(t)
            time_of_day_sec[t]=clocked_sample_times_sec[t_id]
        else:
            time_of_day_sec[t]=time_of_day_sec[t-1]+tstep_sec # simply increment time of day by the time step
    return time_of_day_sec


def prune_annotations(annot_list):
    annot_lower = [annot['text'].lower().strip('\n') for annot in annot_list]

    # Auto-remove some annotations
    rm_events = ['xlspike','xlevent','start recording','video recording on','recording analyzer - xlevent - intracranial',
                 'recording analyzer - xlspike - intracranial','recording analyzer - csa','recording analyzer - ecg',
                 'clip note','started analyzer - xlevent / ecg','started analyzer - csa','started analyzer - xlspike',
                 'persyst - license error','started analyzer - persyst',"please refer to electrode table in patient's folder about correct grid order"]
    pruned_annot1 = []
    pruned_annot1_lower = []
    for ct, annot in enumerate(annot_lower):
        if not (annot in rm_events):
            pruned_annot1_lower.append(annot)
            pruned_annot1.append(annot_list[ct])
    # Ask user to manually select annotations to keep
    # all_done = False
    # n_annot = len(pruned_annot1_lower)
    # while all_done == False:
    #     for ct, annot in enumerate(pruned_annot1_lower):
    #         print('{}: {}'.format(ct, annot))
    #     keep_str = input('Enter the indices of any annotations that should be kept (e.g., 1 2 13 or return for none)')
    #     if len(keep_str) > 0:
    #         keep_ids = [int(str) for str in keep_str.split(' ')]
    #         if np.max(keep_ids) >= n_annot or np.min(keep_ids) < 0:
    #             print('WARNING: Removing ids greater than %d or less than 0' % n_annot)
    #             temp_keep_ids = []
    #             for id in keep_ids:
    #                 if id < n_annot and id >= 0:
    #                     temp_keep_ids.append(id)
    #             keep_ids = list(temp_keep_ids)
    #             del temp_keep_ids
    #     else:
    #         keep_ids = []
    #     if len(keep_ids) == 0:
    #         print('Removing all annotations')
    #     else:
    #         print('Keeping the following annotations:')
    #         for id in keep_ids:
    #             print('{}: {}'.format(id, pruned_annot1_lower[id]))
    #     valid_response = False
    #     while valid_response == False:
    #         double_check = input('Redo or continue (r/c)?')
    #         if double_check.lower() == 'r':
    #             valid_response = True
    #         elif double_check.lower() == 'c':
    #             valid_response = True
    #             all_done = True
    # pruned_annot = [pruned_annot1[id] for id in keep_ids]
    # return pruned_annot
    return pruned_annot1


######## MAIN FUNC ########
def layread(layFileName,datFileName=None,timeOffset=0,timeLength=-1,importDat=True):
    """
    Required Input:
        layFileName - the .lay file name (including path)

    Optional Inputs:
        datFileName - the .dat file name.
        timeOffset - the number of time steps to ignore (so if this was set to 3 for example, the file reader would extract data for time steps 4 to the end)
        timeLength - the number of time steps to read (so if this was set to 5 and timeOffset was set to 3, the file reader would read data for time steps 4,5,6,7,8). If this parameter is set to -1, then the whole .dat file is read.
        importDat - Boolean. If True, the time series data in the dat file are imported

    Default values:
        datFileName - Default: assume the same path and file stem as layFileName
        timeOffset=0 (i.e., start reading that beginning of the file)
        timeLength=-1 (i.e., read in entire file)
        importDat=True

    outputs:
        header - information from .lay file. It contains the following keys:
            samplingrate: sampling rate in Hz
            rawheader: a dict of the raw header from the lay file
            starttime: string indicating the start time of recording (date hours:min:seconds)
            datafile: full path and filename of dat file
            annotations: list of event annotations
            waveformcount: # of channels
            patient: dict of patient information (mostly empty)
        record - EEG data from .dat file (channel x time numpy array)

        Note that the header is the same no matter what length of data are sampled from the dat file.
    """

    # takes ~8 min for a 1.5GB file
    t = time.time()

    # If datFileName not specified, assume it is the same location and has the same stem as layFileName
    if datFileName==None:
        layPath = os.path.dirname(layFileName)
        layFname = os.path.splitext(os.path.basename(layFileName))
        datFileName = os.path.join(layPath, layFname[0] + '.dat')

    # get .ini file and replace [] with ''
    data, sections, subsections = inifile.inifile(layFileName,'readall') # sections and subsections currently unused
    for row in data:
        for entry in row:
            if entry == []:
                entry = ''

    # find fileinfo section of .lay file and map to correct .dat file
    fileInfoArray = []
    for row in data:
        if row[2] == 'file':
            row[3] = datFileName
        if row[0] == 'fileinfo':
            fileInfoArray.append(row)

    # fileinfo
    fileinfo = {} # dictionary
    for row in fileInfoArray:
        fileinfo[row[2]] = row[3]

    # patient
    patient = {} # dictionary
    for row in data:
        if row[0] == 'patient':
            patient[row[2]] = row[3]

    # montage
    montage = {} # dictionary
    for row in data:
        if row[0] == 'montage':
            montage_data = [] # 2d nested list
            for row_ in data:
                if row[2] == row_[0]:
                    montage_data.append([row_[2],row_[3]])
            montage[str(row[2])] = montage_data

    # sampletimes
    sampletimes = [] # list of dictionaries
    for row in data:
        if row[0] == 'sampletimes':
            #print("sample: {}, time: {}".format(float(row[2]),float(row[3])))
            sampletimes.append({'sample':float(row[2]),'time':float(row[3])})
    # Persyst appears to periodically resync the time-sample mapping. sampletimes tells you when those happen.
    # For example, in a short demo files with 256 Hz sampling I get
    # sample: 0.0, time: 34273.799
    # sample: 454540.0, time: 36049.344
    # sample: 907040.0, time: 37816.922
    # Note that time is is units of time of day in seconds. It doesn't cycle to 0 though for recordings that last past
    # midnight; it just continues to increase past 60*60*24. Also note that interval between time samples does not appear
    # to be fixed. In one 24 hr long file I looked at, time stamps were intially about .55 hours apart but then they were
    # about .3 hours apart, with a few values in between.

    # channelmap
    channelmap = [] # list of strings
    for row in data:
        if row[0] == 'channelmap':
            channelmap.append(row[2])

    # move some info from raw header to header
    header = {} # dictionary
    if len(fileInfoArray) > 0:
        # checking individual fields exist before moving them
        if 'file' in fileinfo:
            header['datafile'] = fileinfo['file']
        if 'samplingrate' in fileinfo:
            header['samplingrate'] = int(fileinfo['samplingrate'])
        if 'waveformcount' in fileinfo:
            header['waveformcount'] = int(fileinfo['waveformcount'])
    # NOT IMPLEMENTED dn = datenum(strcat(date, ',', time));
    date = patient['testdate'].replace('.','/')
    tim = patient['testtime'].replace('.',':')
    try:
        #dt = time.strptime(date + ',' + tim,'%m/%d/%y,%H:%M:%S') # TODO this was old code, perhaps format has changed?
        dt = time.strptime(date + ',' + tim, '%Y/%m/%d,%H:%M:%S') #TODO double check
    except:
        raise Exception('Error converting start time to datetime object. Try using the older month-first format.')
    dt = datetime.fromtimestamp(mktime(dt))
    dt = dt.strftime('%d-%b-%Y %H:%M:%S') # convert date and time to standard format
    header['starttime'] = dt
    header['patient'] = patient

    # comments
    try:
        lay_file_ID = open(layFileName,'r')
    except:
        raise Exception('Error in open: file not found')
    comments_ = 0
    cnum = 0
    comments = [] # list of strings
    annotations = [] # list of dictionaries
    for tline in lay_file_ID:
        if 1 == comments_:
            contents = tline.split(',')
            if len(contents) < 5:
                break # there are no more comments
            elif len(contents) > 5:
                separator = ','
                contents[4] = separator.join(contents[4:len(contents)])
            # raw header contains just the original lines
            comments.append(tline.strip()) # These lines look something like this:
            # 127.182,0.000,0,100,XLSpike
            # This first element (in this case 127.182) indicates the time in seconds from the start of the file at which
            # the event occurred
            samplenum = float(contents[0])*float(fileinfo['samplingrate']) # convert onset from seconds to samples
            samplenumRaw=samplenum
            i = 0
            while i < len(sampletimes)-1 and samplenum > sampletimes[i+1]['sample']:
                # i tells you which sample-synchronization point to use in order to map from samples to time
                i=i+1
            samplenum -= sampletimes[i]['sample']
            samplesec = samplenum / float(fileinfo['samplingrate'])
            timesec = samplesec + sampletimes[i]['time']
            commenttime = time.strftime('%H:%M:%S',time.gmtime(timesec)) # should be converted to HH:MM:SS
            dn = patient['testdate'] + ',' + str(commenttime)
            dn = time.strptime(dn,'%Y.%m.%d,%H:%M:%S') # TODO double check, I think also varies from older to newer cases
            dn = datetime.fromtimestamp(mktime(dn))
            dn = dn.strftime('%d-%b-%Y %H:%M:%S') # convert date and time to standard format
            annotations.append({'time':dn, 'sample': int(np.round(samplenumRaw)),'duration':float(contents[1]),'text':contents[4]})
            # annotations[cnum] = {'time':dn} # previously datetime(dn,'ConvertFrom','datenum')
            # annotations[cnum] = {'duration':float(contents[1])}
            # annotations[cnum] = {'text':contents[4]}
            cnum += 1
        elif tline[0:9] == '[Comments]'[0:9]:
            # read until get to comments
            comments_ = 1
    lay_file_ID.close()

    header['annotations'] = annotations # add to header dictionary
    rawhdr = {} # dictionary to represent rawhdr struct
    rawhdr['fileinfo'] = fileinfo
    rawhdr['patient'] = patient
    rawhdr['sampletimes'] = sampletimes
    rawhdr['channelmap'] = channelmap
    rawhdr['comments'] = comments
    rawhdr['montage'] = montage
    header['rawheader'] = rawhdr # put raw header in header

    # dat file
    record=[] # TODO make this an input option
    if importDat:
        try:
            dat_file_ID = open(datFileName,'rb')
        except:
            raise Exception('Error in open: file not found')
        recnum = float(rawhdr['fileinfo']['waveformcount'])
        recnum = int(recnum)
        calibration = float(rawhdr['fileinfo']['calibration'])
        if int(rawhdr['fileinfo']['datatype']) == 7:
            precision = np.int32
            dat_file_ID.seek(recnum*4*timeOffset,1)
        else:
            precision = np.int16
            dat_file_ID.seek(recnum*2*timeOffset,1)

        # read data from .dat file into array of correct size, then calibrate
        # records = recnum rows x inf columns
        if timeLength == -1:
            toRead = -1 # elements of size precision to read
        else:
            toRead = timeLength*recnum
        record = np.fromfile(dat_file_ID,dtype=precision,count=toRead)
        dat_file_ID.close()
        record = record * calibration # explicit
        record = np.reshape(record,(recnum,-1),'F') # recnum rows (i.e., the # of channels)
        record = record.astype(np.float32) # cast as float32; more than enough precision

        # elapsed time (in min)
        elapsed = (time.time() - t) / 60
    else:
        record=np.zeros((0,0))

    return (header,record)

# if __name__ == '__main__':
# 	try:
# 		layread("\Users\Ian\Documents\Adaptive Stimulation\FileReader\skAnonShort.lay","\Users\Ian\Documents\Adaptive Stimulation\FileReader\skAnonShort.dat") # sample lay and dat files i was using
# 	except:
# 		type,value,tb = sys.exc_info()
# 		traceback.print_exc()
# 		pdb.post_mortem(tb)