import numpy as np
import hickle as hkl
from ridge import ridge, ridge_corr, bootstrap_ridge
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tables
from scipy.misc import imresize
import os
from scipy.stats.mstats import zscore

TR = 1.0 #seconds
fps = 15

numfeats = 96*96*3
min_delay = 3 # times TR 

roilist = ['v1lh', 'v1rh']

features_folder = "../../vim2/results"

test_train_cutoff = 1500

frame = 15
featuretest = np.array([])
featuretrain = np.array([])
fname = features_folder + "/errors_frame" + str(frame) + "_" + str(frame +15) +".hkl"
time = 1
print fname
while os.path.isfile(fname):

    if frame > test_train_cutoff:

        if featuretest.size == 0:
            featuretest = np.array([np.mean(hkl.load(fname), axis=0),])
        else:
            #featuretest = np.append(featuretest,np.mean(hkl.load(fname), axis=0))
            featuretest = np.concatenate((featuretest, np.array([np.mean(hkl.load(fname), axis=0),])), axis=0)
    else:

        if featuretrain.size == 0:

            featuretrain = np.array([np.mean(hkl.load(fname), axis=0),])
        else:
            #featuretrain = np.append(featuretrain,np.mean(hkl.load(fname), axis=0))
            featuretrain = np.concatenate((featuretrain, np.array([np.mean(hkl.load(fname), axis=0),])), axis=0)

    frame +=15
    time +=1
    fname = features_folder + "/errors_frame" + str(frame) + "_" + str(frame +15) +".hkl"


"""
features_train = Fi.getNode('/st')
train_frames = features_train.shape[0]


    


features_test = Fi.getNode('/sv')
 
test_frames  = features_test.shape[0]


 
frame = 0
el = 0
featuretrain = np.zeros((train_frames/15, numfeats))
while frame < train_frames:

    chunk = features_train[frame:frame+15] #first resize the image
    chunk.transpose((0,2,3,1))
    resizedchunk = np.zeros((15,96,96,3))
    
    for i in range(15):
        

        resizedchunk[i] = imresize(chunk[i], 96.0/128.0)
        
    featuretrain[el] = np.mean(resizedchunk,axis=0).flatten()
    
    frame += 15
    el +=1
features_train = 0



frame = 0
el = 0
featuretest = np.zeros((test_frames/15, numfeats))
while frame < test_frames:
    chunk = features_test[frame:frame+15] #first resize the image
    chunk.transpose((0,2,3,1))
    resizedchunk = np.zeros((15,96,96,3))
    for i in range(15):
        resizedchunk[i] = imresize(chunk[i], (96,96))
    featuretest[el] = np.mean(resizedchunk,axis=0).flatten()
    
    frame += 15
    el +=1
features_test = 0
"""
print featuretrain.shape
print featuretest.shape

#choose ROI




respfile = tables.openFile("../../vim2/vim-2/VoxelResponses_subject1.mat")

print test_train_cutoff/15
#resptrain = np.transpose(respfile.getNode('/rt')[:][1:])
resptrain = np.transpose(respfile.getNode('/rv')[:,1:int(test_train_cutoff/15.0)])

#resptrain = np.transpose(respfile.get('rt'))

print resptrain.shape

print time
#resptest = np.transpose(respfile.get('rv'))

resptest = np.transpose(respfile.getNode('/rv')[:,int(test_train_cutoff/15.0):time])

Rresp = [] #training data
Presp = [] #test data

print resptest.shape

for roi in roilist:
    if roi == 'all':
       Rresp = resptrain
       Presp = resptest
       break

    roi_idx = np.nonzero(
        respfile.getNode('/roi/'+roi)[:].flatten() == 1)[0]
    if Rresp == []:
        Rresp = resptrain[:,roi_idx]
        Presp = resptest[:,roi_idx]
    else:
        Rresp = np.concatenate((Rresp, resptrain[:,roi_idx]),axis=1)
        Presp = np.concatenate((Presp, resptest[:,roi_idx]),axis=1)

resptest = 0
resptrain = 0

RStim = np.concatenate((np.roll(featuretrain, min_delay), np.roll(featuretrain, min_delay+1), np.roll(featuretrain, min_delay +2)), axis=1)
featuretrain=0

PStim = np.concatenate((np.roll(featuretest, min_delay), np.roll(featuretest, min_delay+1), np.roll(featuretest, min_delay +2)), axis=1)
featuretrain=0

RStim = zscore(RStim, axis=0)
PStim = zscore(PStim, axis =0)

Rresp = zscore(Rresp, axis=0)
Presp = zscore(Presp, axis=0)


corr= ridge_corr(RStim,PStim, Rresp, Presp, [0.1])

hkl.dump(corr, "corr"+".hkl")
plt.hist(corr[0], bins=(200))
plt.savefig("corr.png")
