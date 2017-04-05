import hickle as hkl
from scipy.io import savemat
import numpy as np

"""This code takes the extracted error files and stitches them together chronologically and averages them olve 1 second"""
#15 frames
file_overlap = 10
num_test = 10
test_increment = 840
num_train = 30
train_increment = 540
layer_size = 122880

outtrain= np.zeros((7200, layer_size), dtype=np.float16)
fr = 0
for i in range(num_train):
    part = hkl.load("../vim2/results/trainerr" + str(i) + ".hkl")[:,:layer_size]

    print "train filee no. " + str(i)

    pp =0
    while pp < train_increment:
        outtrain[fr] = np.mean(part[pp:pp+15], axis=0)
        fr +=1
        pp +=15




outtest = np.zeros((540, layer_size))
"""
fr = 0
for i in range(num_test):
    part = hkl.load("../vim2/results/testerr" + str(i) + ".hkl")#[0,:,:layer_size]
    print part[:,0:10,:0:10]
    print "test file no. " + str(i)
    pp = 0
    while pp < test_increment:
        outtest[fr] = np.mean(part[pp:pp+15], axis=0)
        fr += 1
        pp += 15
"""
hkl.dump(outtrain,"../vim2/results/errtrainl1.hkl")
#savemat("../vim2/results/errl1.mat", {"train":outtrain, "test":outtest})
