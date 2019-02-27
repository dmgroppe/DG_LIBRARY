import numpy as np
import os
import shutil
import scipy as sp
import scipy.stats

def ensure_clear_dir(directory):
    """ Checks to see if 'directory' exists. It erases it if does and then creates it. Else it just creates it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # remove the directory
        shutil.rmtree(directory)
        # now create it
        os.makedirs(directory)

def ensure_exists_dir(directory):
    """ Checks to see if 'directory' exists. If not existing, it is created."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def randTpts(nTptsTotal,nTptsShow):
    """ randTpts(nTptsTotal,nTptsShow): Useful for getting a random subset of contiguous time points to plot"""
    showMax=nTptsTotal-nTptsShow
    if showMax<0:
        print('Error: nTptsShow needs to be greater than or equal to nTptsTotal')
        randSubset=np.arange(nTptsTotal) # ?? make this a real error someday
    else:
        randSubset=np.arange(nTptsShow)+np.random.randint(0,showMax)
    return randSubset


def randVals(seq,n=5):
    """ Useful for getting random ids from a sample; ids are NOT contiguous"""
    return seq[np.random.permutation(len(seq))[:n]]


def asin_trans(data, output_degrees=False):
    """ Performs arcsin transformation to make 0-1 range data more normal.
    data should range from 0-1. Output ranges from 0 to pi/2"""
    out_data=np.arcsin(np.sqrt(data))
    if output_degrees==True:
        out_data=out_data*180/np.pi
    return out_data


def find_nearest(array,value):
    """ find_nearest(array,value)
    Returns index of array element that is closest to value."""
    idx = (np.abs(array-value)).argmin()
    return idx


def boxcar_move_avg(x, wind_len, wind_step):
    n=len(x)
    half_wind=int(np.round(wind_len/2))
    cntrs=np.arange(half_wind,n,step)
    n_wind=len(cntrs)
    y=np.zeros(n_wind)
    ctr=0
    for t in range(n_wind):
        y[t] = np.sum(x[ctr:(ctr+N)])
        ctr=ctr+wind_step
    return y/N


def running_mean(x, N):
    """ smoothed_x=running_mean(x, N)
    Applies an N-length boxcar moving average to x.
    smoothed_x will be N-1 points shorter than x """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def be_focused(minutes=15):
    """ Beep after a set # of minutes
    :param minutes: # of minutes before beeping
    :return: nothing is returned
    """
    import time
    time.sleep(minutes*60)
    print("Time's up!")
    for a in range(10):
        print('\a')
    return None


def normalize(data,zero_nans=False, verbose=True):
    """ data should be variables x observations 
        NaN values are ignored when computed mean and standard deviation and then zeroed AFTER data are z-scored.
        
        Inf values are converted to NaNs
        
        NOTE, modifications to data are done in place! Create a copy if you want to keep the original values
        """
    
    # Convert any inf values to NaNs
    data[np.isinf(data)]=np.nan
    
    # Normalize each feature to be zero mean, unit standard deviation
    nDim = data.shape[0]
    nObs = data.shape[1]
    if verbose==True:
        print('{} dimensions'.format(nDim))
        print('{} observations'.format(nObs))
    dataMns = np.zeros(nDim)
    dataSDs = np.zeros(nDim)
    for a in range(nDim):
        # dataMns[a] = np.mean(data[a,:])
        # dataSDs[a] = np.std(data[a,:])
        if sum(np.isnan(data[a,:]))<nObs:
            dataMns[a] = np.nanmean(data[a,:])
            dataSDs[a] = np.nanstd(data[a,:])
            if dataSDs[a]>np.finfo(float).eps:
                # If standard deviation is not too small, divide by it
                data[a,:]=(data[a,:]-dataMns[a])/dataSDs[a]
            else:
                # SD is too small just subtract data mean
                data[a,:]=data[a,:]-dataMns[a]
        if zero_nans==True:
            data[a,np.isnan(data[a,:])]=0
    
    return data, dataMns, dataSDs


def applyNormalize(data,dataMns,dataSDs):
    """ data should be variables x observations
    # Normalize each feature to be zero mean, unit standard deviation,
    # given previously computed means and SD """


    nDim = data.shape
    if nDim[0]!=len(dataMns):
        print('ERROR!!!! data dimensionality not compatible with dataMns')
        # Throw a proper error ??
        return
        
    if nDim[0]!=len(dataSDs):
        print('ERROR!!!! data dimensionality not compatible with dataSDs')
        return
    
    for a in range(nDim[0]):
        if dataSDs[a]>np.finfo(float).eps:
            # If standard deviation is not too small, divide by it
            data[a,:]=(data[a,:]-dataMns[a])/dataSDs[a]
        else:
            # SD is too small just subtract data mean
            data[a,:]=data[a,:]-dataMns[a]

    return data


def trimmed_mn_sd(data,pptn_trim):
    """  mn, sd=trimmed_mn_sd(data,pptn_trim)
    returns trimmed mean and standard deviation
    (i.e., sample size*(1-2*pptn_trim) of the samples will be used to compute mean and sd.
    nan values are ignored

    data should be a vector
    """
    n_obs=len(data)

    # number of observations to remove from top and bottom
    n_trim=int(np.floor(pptn_trim*n_obs))

    # remove nan values
    use_data=data[~np.isnan(data)]

    # sort data
    use_data.sort()

    if n_trim==0:
        # compute mean
        mn=np.mean(use_data)
        # compute sd with slicing
        sd = np.std(use_data)
    elif n_trim*2>=n_obs:
        raise Warning("pptn_trim needs to be less than 0.5")
        mn=np.nan()
        sd=np.nan()
    else:
        # compute mean
        mn=np.mean(use_data[n_trim:-n_trim])
        # compute sd with slicing
        sd = np.std(use_data[n_trim:-n_trim])
    return mn, sd


def trimmed_normalize(data, pptn_trim, zero_nans=False, verbose=True):
    """ data should be variables x observations
        NaN values are ignored when computed mean and standard deviation and then zeroed AFTER data are z-scored.
        (i.e., sample size*(1-2*pptn_trim) of the samples will be used to compute mean and sd.
        Inf values are converted to NaNs

        NOTE, modifications to data are done in place! Create a copy if you want to keep the original values
        """

    # Convert any inf values to NaNs
    data[np.isinf(data)] = np.nan

    # Normalize each feature to be zero mean, unit standard deviation
    nDim = data.shape[0]
    nObs = data.shape[1]
    if verbose==True:
        print('{} dimensions'.format(nDim))
        print('{} observations'.format(nObs))
    dataMns = np.zeros(nDim)
    dataSDs = np.zeros(nDim)
    for a in range(nDim):
        if sum(np.isnan(data[a, :])) < nObs:
            dataMns[a], dataSDs[a]=trimmed_mn_sd(data[a, :], pptn_trim)
            if dataSDs[a] > np.finfo(float).eps:
                # If standard deviation is not too small, divide by it
                data[a, :] = (data[a, :] - dataMns[a]) / dataSDs[a]
            else:
                # SD is too small just subtract data mean
                data[a, :] = data[a, :] - dataMns[a]
        if zero_nans == True:
            data[a, np.isnan(data[a, :])] = 0

    return dataMns, dataSDs


def median_normalize(data, zero_nans=False, verbose=True):
    """ data should be variables x observations
        NaN values are ignored when computed mean and standard deviation and then zeroed AFTER data are z-scored.
        (i.e., sample size*(1-2*pptn_trim) of the samples will be used to compute mean and sd.
        Inf values are converted to NaNs

        NOTE, modifications to data are done in place! Create a copy if you want to keep the original values
        """

    # Convert any inf values to NaNs
    data[np.isinf(data)] = np.nan

    # Substract the median of each feature and divide by IQR
    nDim = data.shape[0]
    nObs = data.shape[1]
    if verbose==True:
        print('{} dimensions'.format(nDim))
        print('{} observations'.format(nObs))
    dataMds = np.zeros(nDim)
    dataIQRs = np.zeros(nDim)
    for a in range(nDim):
        if sum(np.isnan(data[a, :])) < nObs:
            dataMds[a]=np.median(data[a, :])
            dataIQRs[a]=sp.stats.iqr(data[a, :])
            if dataIQRs[a] > np.finfo(float).eps:
                # If IQR is not too small, divide by it
                data[a, :] = (data[a, :] - dataMds[a]) / dataIQRs[a]
            else:
                # IQR is too small just subtract data median
                data[a, :] = data[a, :] - dataMds[a]
        if zero_nans == True:
            data[a, np.isnan(data[a, :])] = 0

    return dataMds, dataIQRs

def sphereCntrData(X,pvaKeep=1,epsilon=1E-18):
    """ Add comments??
    # Matrix X should be variables x observations """
    
    # Note that the inverse of the whitening matrix is simply its transpose
    # Epsilon is small value that helps prevent the inverse of small eigenvalues from being huge

    # ?? Add option to sample a subset of data (either randomly or at uniform intervals)
    
    # De-mean the data, must be a faster way to do this ??
    nVar, nObs=X.shape
    mnsX=np.zeros(nVar)
    for a in range(nVar):
        mnsX[a]=np.mean(X[a,:])
        X[a,:]=X[a,:]-mnsX[a]
    
    # Estimate of covariance matrix
    Xcov = np.dot(X,X.T)/(nObs-1)
    
    # Eigenvalue decomposition of the covariance matrix
    eigVals, eigVecs = np.linalg.eigh(Xcov) # eigenvectors are columns, rightmost column has biggest eigenvalue
    pva=eigVals/sum(eigVals) # proportion of variance accounted for by each eigenvector
    #pva=pva[::-1] # reverse order

    # Compute cumulative variance accounted for
    if pvaKeep<1:
        cumCt=nVar-1
        cumPva=pva[cumCt]
        while cumPva<pvaKeep:
            cumCt-=1
            cumPva+=pva[cumCt]
        nKeep=nVar-cumCt  #need to add 2 since a starts at 0
    else:
        nKeep=nVar
        cumPva=1
    print('Using %d of %d PCs to capture %f pcnt of variance' %  (nKeep, nVar, cumPva*100))

    # Diagonal matrix of sqrt of inverted eigenvalues
    D = np.diag(1. / np.sqrt(eigVals+epsilon))
    #D = np.diag(1. / np.sqrt(eigVals[-nKeep:]+epsilon))

    # Whitening matrix
    #W=np.dot(D[:,-nKeep:],eigVecs[:,-nKeep:].T)
    #W=np.dot(D,eigVecs[:,-nKeep:].T)
    W=np.dot(D,eigVecs.T)
    invW=np.linalg.inv(W)
    W=W[-nKeep:,:]
    invW=invW[:,-nKeep:]

    # Multiply by the whitening matrix
    sphereX = np.dot(W,X)

    return sphereX, W, invW, mnsX


def applySphereCntr(data,dataSphere,dataMns):
    """ ?? comments """

    nVar, nObs=data.shape
    # ?? check that sphere matrix and mns shapes are compatible

    # De-Mean the Data
    for a in range(nVar):
        data[a,:]-=dataMns[a]

    # Apply sphering matrix
    data=np.dot(dataSphere,data)

    return data


def invSphereCntr(data,invDataSphere,dataMns):
    """ ?? comments """
    
    nVar, nObs=data.shape
    # ?? check that sphere matrix and mns shapes are compatible
    
    # Apply inverse of sphering matrix
    data=np.dot(invDataSphere,data)
    
    # Re-Mean the Data
    for a in range(nVar):
        data[a,:]+=dataMns[a]

    return data


def pcaCumVA(X):
    """ Add comments??
    Matrix X should be variables x observations """
    
    # Note that the inverse of the whitening matrix is simply its transpose
    # Epsilon is small value that helps prevent the inverse of small eigenvalues from being huge
    
    # ?? Add option to sample a subset of data (either randomly or at uniform intervals)
    
    # De-mean the data, must be a faster way to do this ??
    nVar, nObs=X.shape
    mnsX=np.zeros(nVar)
    for a in range(nVar):
        mnsX[a]=np.mean(X[a,:])
        X[a,:]=X[a,:]-mnsX[a]

    # Estimate of covariance matrix
    Xcov = np.dot(X,X.T)/(nObs-1)
    
    # Eigenvalue decomposition of the covariance matrix
    eigVals, eigVecs = np.linalg.eigh(Xcov) # eigenvectors are columns, rightmost column has biggest eigenvalue
    pva=eigVals/sum(eigVals) # proportion of variance accounted for by each eigenvector
    #pva=pva[::-1] # reverse order

    cumVA=np.zeros(nVar)
    cumVA[0]=pva[nVar-1]
    for a in range(nVar-1):
        cumVA[a+1]=cumVA[a]+pva[nVar-a-2]

    return cumVA


def mean_and_cis(data, confidence=0.95,prune_nan=False):
    """ Returns mean and 95% confidence intervals (normally distributed data assumed) """
    if prune_nan==True:
        data=data[~np.isnan(data)]
    n = len(data)
    m = np.mean(data)
    se = np.std(data)/np.sqrt(n)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h