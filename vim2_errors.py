'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h
import hickle as hkl
import theano
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
import time
import sys
from prednet import PredNet
#from data_utils import SequenceGenerator 

starttime = time.time() 

WEIGHTS_DIR = "model_data"
DATA_DIR = "../vim2/preprocessed"
RESULTS_SAVE_DIR = "../vim2/results/"

n_plot = 40
batch_size = 15
fperbat = 810

nt = fperbat + 10
tot_frames = 8100
nbat = tot_frames/fperbat

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = DATA_DIR+ '/vim2_test'
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'all_error'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)


X_test = np.zeros((1,8100,3,128,160))
X_test[0,:,:,:,:] = hkl.load("../vim2/preprocessed/test.hkl").transpose((0,3,1,2))

print X_test.shape
print "????"
frame = 10
for i in range(nbat):
    if time.time() - starttime > 72000:
        break
    if frame + fperbat >= X_test.shape[1]:
        frame = X_test.shape[1] - fperbat
    test_errors = test_model.predict(X_test[:,frame-10:frame+fperbat,:,:,:], 1)
    outfile = RESULTS_SAVE_DIR + "testerr"+str(i)+ ".hkl"
    print outfile
    frame += fperbat
    hkl.dump(test_errors[0,9:], outfile)
"""
#    hkl.dump(errs1[:,9,:], RESULTS_SAVE_DIR
#+ "errors_frame"+ str(b*batch_size+9+6) + "_" + str((b+1)*batch_size+9+6)
#+".hkl")

#X_hat = test_model.predict(X_test[1], batch_size)
#test_model._make_predict_function()
#f = test_model.predict_function
#errs1 = f(X_test[0])

# 





nframes = 108000
fperfile = 5400
fperbat = 540
nbatches = nframes/fperbat
file_overlap = 10

X_train = np.zeros((1,fperfile+file_overlap, 3,128,160))
#X_train[0,file_overlap:,:,:,:] = hkl.load("../vim2/preprocessed/train0_5400.hkl").transpose((0,3,1,2))
leftover = np.zeros((file_overlap, 3, 128, 160))
frame = file_overlap
print X_train.shape
for i in range(nbatches):
    if time.time() - starttime > 72000:
        break

    indexstart = frame % fperfile #position in file
    if indexstart == file_overlap:
        #rework x_test
        fname = "../vim2/preprocessed/train" + str(frame-file_overlap) + "_" + str(frame-file_overlap + fperfile) + ".hkl"
        print "loading file " + fname



        leftover = X_train[0,-file_overlap:,:,:,:]
        X_train[:,:file_overlap,:,:,:] = leftover
        
        X_train[:,file_overlap:,:,:,:] = hkl.load(fname).transpose((0,3,1,2))
    print X_train.shape
    errs = test_model.predict(X_train[:,frame-file_overlap:frame+fperbat,
                                      :,:,:])
    hkl.dump(errs[0,file_overlap:],"../vim2/results/trainerr" + str(i)+".hkl")
"""







