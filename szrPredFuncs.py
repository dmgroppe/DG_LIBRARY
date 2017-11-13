import os
import pandas as pd
import pickle
import numpy as np
import sys

workingDirectory = os.getcwd()
if 'HoneyLab' in workingDirectory:
    mach='iMac'
elif 'davidgroppe' in workingDirectory:
    mach = 'lappy'
elif '/h/honey/' in workingDirectory:
    mach = 'scinet'
else:
    mach='hongkong'
print('szrPredFuncs.py: Computer is of type {}'.format(mach))

# Import DG_LIBRARY
if mach=='scinet':
    sys.path.append('/home/h/honey/dgroppe/DG_LIBRARY')
elif mach=='lappy':
    sys.path.append('/Users/davidgroppe/PycharmProjects/DG_LIBRARY')
elif mach=='iMac':
    sys.path.append('/Users/HoneyLab/PycharmProjects/SZR_PRED/DG_LIBRARY')
else:
    #hongkong
    sys.path.append('/home/dgroppe/GIT/DG_LIBRARY')
import dgFuncs as dg

# Global Params
dropThresh = 0.95


def saveKerasModel(model,modelFile,modelWtsFile):
    # Save the model
    # You need to save sequential models this way as they do not have a save attribute (e.g., model.save('myModel.hdf5')
    json_string = model.to_json()
    if os.path.isfile(modelFile):
        os.remove(modelFile) # Remove previous file
    print('Saving model to file {}'.format(modelFile))
    open(modelFile, 'w').write(json_string)
    if os.path.isfile(modelWtsFile):
        os.remove(modelWtsFile) # Remove previous file
    print('Saving model weights to file {}'.format(modelWtsFile))
    model.save_weights(modelWtsFile)

def getCodeDir(machine):
    if machine=='lappy':
        codeDir='/Users/davidgroppe/PycharmProjects/PRED_ICTAL'
    elif machine=='iMac':
        codeDir='/Users/HoneyLab/PycharmProjects/SZR_PRED/PRED_ICTAL'
    elif machine=='scinet':
        codeDir='/scratch/h/honey/dgroppe/PRED_ICTAL'
    else:
        #hong kong
        codeDir = '/home/dgroppe/GIT/PRED_ICTAL'
    return codeDir

def getCodeDirV5(machine):
    if machine=='lappy':
        codeDir='/Users/davidgroppe/PycharmProjects/PRED_PRE_LSTM'
    elif machine=='iMac':
        codeDir='/Users/HoneyLab/PycharmProjects/SZR_PRED/PRED_PRE_LSTM'
    elif machine=='scinet':
        codeDir='/scratch/h/honey/dgroppe/PRED_PRE_LSTM'
    else:
        #hong kong
        codeDir = '/home/dgroppe/GIT/PRED_PRE_LSTM'
    return codeDir

def getDataDir(machine):
    if machine=='lappy':
        dataDir='/Users/davidgroppe/ONGOING/SZR_PRED/'
    elif machine=='iMac':
        dataDir='/Users/HoneyLab/PycharmProjects/SZR_PRED/'
    elif machine=='scinet':
        dataDir='/scratch/h/honey/dgroppe/'
    else:
        #hong kong
        #dataDir = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/'
        dataDir = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/'
    return dataDir


def genCrossValidSets(sub,seed=-1):
    """ genCrossValidSets(seed=-1)
    This function loads the training data data frame, trainDf.pkl, and randomly generates a new division of 10 min
    datasets into training, testing, & validation subsets for one subject.

    It returns the dataframe tempDf
    """
    codeDir=getCodeDir(mach)

    # Seed random number generator
    if seed>=0:
        print('Seeding rand # generator with {}'.format(seed))
        np.random.seed(seed)
    else:
        print('Randomly dividing data into train, valid, and test subsets.')

    # Load training data data frame
    if mach=='scinet':
        trainDf = pickle.load(open(os.path.join(codeDir, 'DATA_INFO', 'trainDfPy2.pkl'), 'rb'))
    else:
        trainDf = pickle.load(open(os.path.join(codeDir,'DATA_INFO','trainDf.pkl'), 'rb'))

    # Get subset of data frame for just this subject
    tempDf = trainDf[trainDf['subId'] == sub]
    tempDf.index = range(tempDf.shape[0])

    # Count the # of inter and preictal data epochs for this sub
    nInter=sum(tempDf['ictalType']=='inter')
    nPre = sum(tempDf['ictalType'] == 'pre')

    # Interictal Data
    nInterHours = int(nInter / 6)
    nValidInter = int(np.round(nInterHours * .2))
    nTrainInter = int(nInterHours - nValidInter)
    print('Total Interictal Hours: {}'.format(nInterHours))
    print('nValid Interictal: {}'.format(nValidInter))
    print('nTrain Interictal: {}'.format(nTrainInter))

    permInter = np.random.permutation(nInterHours)
    validIdsInter=permInter[:nValidInter]
    trainIdsInter=permInter[nValidInter:]
    # Double check that all hours were assigned to one and only subset
    # dude = np.concatenate((validIdsInter[subLoop], trainIdsInter[subLoop]))
    # print(np.sort(dude)[:50])
    # print(len(np.unique(dude)))

    # Preictal Data
    nPreHours = int(nPre / 6)
    nValidPre = int(np.round(nPreHours * .16))
    nTrainPre = int(nPreHours - nValidPre)
    # print('Total Preictal Hours: {}'.format(nPreHours))
    # print('nValid Preictal: {}'.format(nValidPre))
    # print('nTrain Preictal: {}'.format(nTrainPre))

    permPre = np.random.permutation(nPreHours)
    validIdsPre=permPre[:nValidPre]
    trainIdsPre=permPre[nValidPre:]
    # print(validIdsPre[subLoop])
    # print(testIdsPre[subLoop])
    # print(trainIdsPre[subLoop])
    # Double check that all hours were assigned to one and only subset
    # dude=np.concatenate((validIdsPre[subLoop],testIdsPre[subLoop],trainIdsPre[subLoop]))
    # print(np.sort(dude))
    # print(len(np.unique(dude)))
    # print()
    for ct in range(tempDf.shape[0]):
        #if tempDf.loc[ct, 'ictalType'] == 'inter':
        if tempDf.get_value(ct,'ictalType') == 'inter':
            if tempDf['hourGroup'][ct] in validIdsInter:
                tempDf.set_value(ct, 'xvalSubset','valid')
            else:
                tempDf.set_value(ct, 'xvalSubset', 'train')
        else:
            if tempDf['hourGroup'][ct] in validIdsPre:
                tempDf.set_value(ct, 'xvalSubset', 'valid')
            else:
                tempDf.set_value(ct, 'xvalSubset', 'train')

    return tempDf



def getFileList(sub, dataType='all', usePkl=True):
    """
    Returns a list of data files of a particular type

    getFileList(sub, dataType='all', usePkl=True)
    :param sub: integer 1-3
    :param dataType: can be 'all', 'train', or 'test' {default='all'}
    :param usePkl: boolean, if true, "zeroed" data in pkl format are returned. Otherwise original mat files are looked for
    {default='true'}
    :return: a comprehensive list of the specified file types (including path)
    """

    workingDirectory = os.getcwd()

    fileList = []
    if 'HoneyLab' in workingDirectory:
        rootPath ='/Users/HoneyLab/PycharmProjects/SZR_PRED/PKL_DATA/'
    elif 'davidgroppe' in workingDirectory:
        # Lappy
        rootPath = '/Users/davidgroppe/ONGOING/SZR_PRED/PKL_DATA/'
    else:
        # Hong Kong
        rootPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/PKL_DATA/' #TODO this folder no exist yet

    if dataType == 'all' or dataType == 'train':
        # Collect training files (mostly interictal)
        if usePkl:
            inDir = 'train' + '_' + str(sub) + '_zeroed'
        else:
            inDir = 'train' + '_' + str(sub)
        dataPath = rootPath + inDir
        print('Collecting filenames from: {}'.format(dataPath))
        trainFileList = os.listdir(dataPath)

        for f in trainFileList:
            fileList.append(os.path.join(dataPath, f))

    if dataType == 'all' or dataType == 'test':
        # Collect testing files
        if usePkl:
            inDir = 'test' + '_' + str(sub) + '_zeroed'
        else:
            inDir = 'test' + '_' + str(sub)
        dataPath = rootPath + inDir
        print('Collecting filenames from: {}'.format(dataPath))
        testFileList = os.listdir(dataPath)

        for f in testFileList:
            fileList.append(os.path.join(dataPath, f))

    return fileList


def importTrainData(sub,ftr='psd',using_tf=False):
    """ importPsdData(sub,codeDir,norm='sphere'):
    norm can be 'sphere' or 'zscore'
    returns:
    p """

    codeDir=getCodeDir(mach)
    if mach=='scinet':
        trainDfFname = os.path.join(codeDir, 'DATA_INFO', 'trainDfPy2.pkl')
    else:
        trainDfFname=os.path.join(codeDir,'DATA_INFO','trainDf.pkl')
    trainDf=pickle.load(open(trainDfFname,'rb'))

    #### Collect Training Data
    subOnlyDf = trainDf[trainDf['subId'] == sub]
    nTrainFiles = subOnlyDf.shape[0]
    print('# of training 10 min files for Sub {}: {}'.format(sub, nTrainFiles))

    if ftr!='psd':
        print('importTrainData only works for psd currently')
        return

    # Load Data
    nDim = 128
    nWind = 598
    # dropThresh = 0.95

    if 'davidgroppe' in codeDir:
        # Laptop
        psdPath = '/Users/davidgroppe/ONGOING/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/davidgroppe/ONGOING/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    elif 'HoneyLab' in codeDir:
        # iMac
        psdPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    else:
        # Hong Kong
        psdPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/DROP_FEATURES/train_' + str(sub) + '_drop/'


    fnameList=[]
    if using_tf==True:
        print('Importing data according to TensorFlow format')
        dataClass=np.zeros((nWind*nTrainFiles,1))
    else:
        print('Importing data according to Keras format')
        dataClass = np.zeros(nWind * nTrainFiles)
    hourGroups=np.zeros(nWind*nTrainFiles)
    clipIds=np.zeros(nWind*nTrainFiles)
    data=np.zeros((nDim,nWind*nTrainFiles))
    windCt=0
    for a in range(nTrainFiles):
        rootFname = subOnlyDf.values[a][1].split('.')[0]
        dropFname = rootFname + '_drop.pkl'
        dropData = pickle.load(open(os.path.join(dropPath, dropFname), 'rb'))
        subThreshBool = dropData < dropThresh
        nSubThresh = sum(subThreshBool)
        # hourGroup = subOnlyDf.values[a][4]

        # if data doesn't consist soley of dropout, import it:
        if nSubThresh > 0:
            ictalClass = rootFname.split('_')[-1]
            for repLoop in range(nSubThresh):
                fnameList.append(rootFname)
            hourGroups[windCt:windCt+nSubThresh]=subOnlyDf.values[a][4]
            clipIds[windCt:windCt+nSubThresh]=a
            if ictalClass=='1':
                if using_tf == True:
                    dataClass[windCt:windCt + nSubThresh,:] = 1
                else:
                    dataClass[windCt:windCt + nSubThresh] = 1

            psdFname = rootFname + '_psd.pkl'
            tempPsdInputs = pickle.load(open(os.path.join(psdPath, psdFname), 'rb'))
            data[:,windCt:windCt + nSubThresh]=tempPsdInputs[:, subThreshBool]
            windCt+=nSubThresh

    # Release unused preallocated memory
    data=data[:,:windCt]
    if using_tf == True:
        dataClass=dataClass[:windCt,:]
    else:
        dataClass=dataClass[:windCt]
    hourGroups=hourGroups[:windCt]
    clipIds = clipIds[:windCt]
    del tempPsdInputs
    return data, dataClass, hourGroups, clipIds, fnameList


def importTrainLists(sub,ftr='psd'):
    """ importPsdData(sub,codeDir,norm='sphere'):
    norm can be 'sphere' or 'zscore'
    returns:
    p """

    codeDir=getCodeDir(mach)
    if mach=='scinet':
        trainDfFname = os.path.join(codeDir, 'DATA_INFO', 'trainDfPy2.pkl')
    else:
        trainDfFname=os.path.join(codeDir,'DATA_INFO','trainDf.pkl')
    trainDf=pickle.load(open(trainDfFname,'rb'))

    #### Collect Training Data
    subOnlyDf = trainDf[trainDf['subId'] == sub]
    nTrainFiles = subOnlyDf.shape[0]
    print('# of training 10 min files for Sub {}: {}'.format(sub, nTrainFiles))

    if ftr!='psd':
        print('importTrainData only works for psd currently')
        return

    # Load Data
    nDim = 128
    nWind = 598
    # dropThresh = 0.95

    if 'davidgroppe' in codeDir:
        # Laptop
        psdPath = '/Users/davidgroppe/ONGOING/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/davidgroppe/ONGOING/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    elif 'HoneyLab' in codeDir:
        # iMac
        psdPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    else:
        # Hong Kong
        psdPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/DROP_FEATURES/train_' + str(sub) + '_drop/'

    fnameList=[]

    hourGroups=np.zeros(nWind*nTrainFiles)
    clipIds=np.zeros(nWind*nTrainFiles)
    data=np.zeros((nDim,nWind*nTrainFiles))
    windCt=0
    for a in range(nTrainFiles):
        rootFname = subOnlyDf.values[a][1].split('.')[0]
        dropFname = rootFname + '_drop.pkl'
        dropData = pickle.load(open(os.path.join(dropPath, dropFname), 'rb'))
        subThreshBool = dropData < dropThresh
        nSubThresh = sum(subThreshBool)
        # hourGroup = subOnlyDf.values[a][4]

        # if data doesn't consist soley of dropout, import it:
        if nSubThresh > 0:
            ictalClass = rootFname.split('_')[-1]
            for repLoop in range(nSubThresh):
                fnameList.append(rootFname)
            hourGroups[windCt:windCt+nSubThresh]=subOnlyDf.values[a][4]
            clipIds[windCt:windCt+nSubThresh]=a
            if ictalClass=='1':
                if using_tf == True:
                    dataClass[windCt:windCt + nSubThresh,:] = 1
                else:
                    dataClass[windCt:windCt + nSubThresh] = 1

            psdFname = rootFname + '_psd.pkl'
            tempPsdInputs = pickle.load(open(os.path.join(psdPath, psdFname), 'rb'))
            data[:,windCt:windCt + nSubThresh]=tempPsdInputs[:, subThreshBool]
            windCt+=nSubThresh

    # Release unused preallocated memory
    data=data[:,:windCt]
    if using_tf == True:
        dataClass=dataClass[:windCt,:]
    else:
        dataClass=dataClass[:windCt]
    hourGroups=hourGroups[:windCt]
    clipIds = clipIds[:windCt]
    del tempPsdInputs
    return data, dataClass, hourGroups, clipIds, fnameList



def importPsdTestData(sub, modelName, xvalRep, norm):
    """ Imports and normalizes test data for a subject"""

    # import sample_submission.csv to get list of all the filenames that should be there
    # mach is set at top of file
    if mach == 'lappy':
        codePathStem = '/Users/davidgroppe/PycharmProjects/PRED_ICTAL/'
        dataDir = '/Users/davidgroppe/ONGOING/SZR_PRED/'
    elif mach == 'hongkong':
        codePathStem = '/home/dgroppe/GIT/PRED_ICTAL/'
        dataDir = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/'
    else:
        # Honeylab iMac
        codePathStem = '/Users/HoneyLab/PycharmProjects/SZR_PRED/PRED_ICTAL/'
        dataDir = '/Users/HoneyLab/PycharmProjects/SZR_PRED/'

    sampleSubmitCsv = os.path.join(codePathStem,'PHAT','sample_submission.csv')
    sampleSubmitDf = pd.read_csv(sampleSubmitCsv)

    # Get list of matFiles for just this sub:
    subMatFiles = []
    nTestFilesAllSubs=sampleSubmitDf.shape[0]
    for a in range(nTestFilesAllSubs):
        if int(sampleSubmitDf['File'].values[a][0]) == sub:
            subMatFiles.append(sampleSubmitDf['File'].values[a])
    nTestFiles=len(subMatFiles)

    # Import list that indicates which files have complete dropout
    testDropoutFname = os.path.join(codePathStem, 'DATA_INFO', 'testDropout.pkl')
    testDropout = pickle.load(open(testDropoutFname, 'rb'))
    # testDropout is a dict with keys: ['files1', 'nDropout3', 'nDropout1', 'files3', 'nDropout2', 'files2']
    nTimePts = 240000 # The number of time points in each raw data file. You need this to recognize which files in
    # testDropout have complete dropout

    # Get list of psd data files and make sure we have one for every file that does not have complete dropout
    psdDir = os.path.join(dataDir, 'PSD_FEATURES', 'test_' + str(sub) + '_psd')
    testPsdFileList = os.listdir(psdDir)

    matFilePath = os.path.join(dataDir,'SZR_PRED_ORIG_DATA','test_' + str(sub))
    missingPklFile = []
    noData=np.zeros(nTestFiles,dtype='int16')
    ct=0
    for matF in subMatFiles:
        # Check to see if that mat file had complete dropout
        matId = testDropout['files' + str(sub)].index(matF)
        nDropThisFile = testDropout['nDropout' + str(sub)][matId]
        if nDropThisFile < nTimePts:
            # We should have a psd file for that mat file then
            psdFile = matF.split('.')[0] + '_psd.pkl'
            if psdFile in testPsdFileList:
                pass
            else:
                missingPklFile.append(os.path.join(matFilePath, matF))
                print('Could not find file {} for test file {} (nDrop={}, pDrop={:.3f})'.format(psdFile,
                                                                                                matF, nDropThisFile,
                                                                                                nDropThisFile / nTimePts))
        else:
            noData[ct]=1
        ct+=1

    if len(missingPklFile) > 0:
        print('Missing psd feature files for Sub{}!'.format(sub))
        print('Saving list of mat files with missing psd files to tempMissedFiles.pkl')
        pickle.dump(missingPklFile, open('tempMissedFiles.pkl', 'wb'))

    # Load normalization info
    if norm=='sphere':
        xValDir=os.path.join(codePathStem,'MODELS',modelName,'SUB'+str(sub),'XVAL_MODELS')
        mnsFname=os.path.join(xValDir,'sub'+ str(sub) + '_Mns_' + str(xvalRep) + '.pkl')
        mnsInputs=pickle.load(open(mnsFname,'rb'))
        wFname = os.path.join(xValDir,'sub'+ str(sub) + '_W_' + str(xvalRep) + '.pkl')
        W=pickle.load(open(wFname,'rb'))
    else:
        xValDir = os.path.join(codePathStem, 'MODELS', modelName, 'SUB' + str(sub), 'XVAL_MODELS')
        mnsFname = os.path.join(xValDir, 'sub' + str(sub) + '_Mns_' + str(xvalRep) + '.pkl')
        mnsInputs = pickle.load(open(mnsFname, 'rb'))
        sdsFname = os.path.join(xValDir, 'sub' + str(sub) + '_SDs_' + str(xvalRep) + '.pkl')
        mnsSDs = pickle.load(open(sdsFname, 'rb'))

    # Loop over test files
    psdTestInputList=[]
    for fileLoop in range(nTestFiles):
        fileStem=subMatFiles[fileLoop].split('.')[0]

        # If file has data import psd data
        if noData[fileLoop]==0:
            # Import array of proportion dropout per time window
            pDropFname = os.path.join(dataDir, 'DROP_FEATURES', 'test_'+str(sub)+'_drop', fileStem + '_drop.pkl')
            #print('Loading: {}'.format(pDropFname))
            dropData = pickle.load(open(pDropFname, 'rb'))

            psdFname = os.path.join(dataDir, 'PSD_FEATURES', 'test_'+str(sub)+'_psd', fileStem + '_psd.pkl')
            #print('Loading: {}'.format(psdFname))
            tempPsdInputs = pickle.load(open(psdFname, 'rb'))

            # Remove time windows without data
            subThreshBool=dropData<dropThresh
            tempPsdInputs = tempPsdInputs[:, subThreshBool]

            # Normalize data
            if norm == 'sphere':
                # Sphere inputs
                psdTestInputList.append(dg.applySphereCntr(tempPsdInputs, W, mnsInputs))
            else:
                # zscore inputs
                psdTestInputList.append(dg.applyNormalize(tempPsdInputs, mnsInputs, mnsSDs))

        else:
            psdTestInputList.append([]) #Indicates that file consists solely of dropout

    # Return data, list of files with complete dropout and array of pptn of dropout points in each moving time window
    return psdTestInputList, subMatFiles


def importPsdDataNoXval(sub,codeDir,norm='sphere'):
    """ importPsdData(sub,codeDir,norm='sphere'):
    norm can be 'sphere' or 'zscore'
    returns:
    psdTestInputList, testClasses, psdValidInputList, validClasses, psdTrainInputs, trainClasses,  mnsInputs, mnsSDs, W """

    # Create training data dataframe that randomly assigns data to train, valid, and test subsets
    trainDf=genCrossValidSets(codeDir)

    #### Collect ALL Training Data
    trainOnlyDf = trainDf[trainDf['subId'] == sub]
    nTrainFiles = trainOnlyDf.shape[0]
    print('# of training 10 min files for Sub {}: {}'.format(sub, nTrainFiles))

    # Load Data
    nDim = 128
    nWind = 598
    # dropThresh = 0.95
    psdTrainInputs = np.zeros((nDim, nWind * nTrainFiles))
    trainClasses = np.zeros(nWind * nTrainFiles, dtype='int32')  # TODO use bool dtype??
    # psdPath = '/Users/davidgroppe/ONGOING/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
    # dropPath = '/Users/davidgroppe/ONGOING/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    if 'davidgroppe' in codeDir:
        # Laptop
        psdPath = '/Users/davidgroppe/ONGOING/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/davidgroppe/ONGOING/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    elif 'HoneyLab' in codeDir:
        # iMac
        psdPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/Users/HoneyLab/PycharmProjects/SZR_PRED/DROP_FEATURES/train_' + str(sub) + '_drop/'
    else:
        # Hong Kong
        psdPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/PSD_FEATURES/train_' + str(sub) + '_psd/'
        dropPath = '/media/dgroppe/a6113fdf-562d-3faa-8fa9-91cdd849cb10/SZR_PREDICTION_2016/DROP_FEATURES/train_' + str(sub) + '_drop/'
    ct = 0

    for a in range(nTrainFiles):
        rootFname = trainOnlyDf.values[a][1].split('.')[0]
        dropFname = rootFname + '_drop.pkl'
        dropData = pickle.load(open(os.path.join(dropPath, dropFname), 'rb'))
        subThreshBool = dropData < dropThresh
        nSubThresh = sum(subThreshBool)

        # if data doesn't consist soley of dropout, import it:
        if nSubThresh > 0:
            ictalClass = rootFname.split('_')[-1]
            if ictalClass == '1':
                # preictal data
                trainClasses[ct:(ct + nSubThresh)] = 1

            psdFname = rootFname + '_psd.pkl'
            tempPsdInputs = pickle.load(open(os.path.join(psdPath, psdFname), 'rb'))
            # only import time windows with less than subThresBool dropout
            psdTrainInputs[:, ct:(ct + nSubThresh)] = tempPsdInputs[:, subThreshBool]
            # # CHEAT
            # if ictalClass == '1':
            #     psdTrainInputs[:, ct:(ct + nSubThresh)]=np.ones((nDim,nSubThresh))
            # else:
            #     psdTrainInputs[:, ct:(ct + nSubThresh)] = np.zeros((nDim, nSubThresh))
            ct += nSubThresh

    # Remove unused pre-allocated memory
    psdTrainInputs = psdTrainInputs[:, :ct]
    trainClasses = trainClasses[:ct]
    #print('# of time windows: {}'.format(ct))

    # Normalize Data
    if norm=='sphere':
        print('Normalizing sensor data via sphering')
        pvaKeep = .999
        psdTrainInputs, W, invW, mnsInputs = dg.sphereCntrData(psdTrainInputs, pvaKeep)
        mnsSDs=None
    else:
        print('Normalizing sensor data via z-scoring')
        psdTrainInputs, mnsInputs, mnsSDs = dg.normalize(psdTrainInputs)
        W=None

    return psdTrainInputs, trainClasses, mnsInputs, mnsSDs, W


def roc_auc(ctgry, pHat):
    """ Computes area under the ROC curve for an array of binary class values, ctgry, and a
     an array of binary class probabilities. """
    uniP = np.unique(pHat)
    nUni = len(uniP)
    fPosRate = np.zeros(nUni)
    tPosRate = np.zeros(nUni)
    posBool = (ctgry == 1)
    negBool = (ctgry == 0)
    for a in range(nUni):
        tPosRate[a] = np.mean(pHat[posBool] >= uniP[a])
        fPosRate[a] = np.mean(pHat[negBool] >= uniP[a])

    auc = -np.trapz(tPosRate, fPosRate)
    return (auc)