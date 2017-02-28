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

from prednet import PredNet
#from data_utils import SequenceGenerator 

starttime = time.time() 

WEIGHTS_DIR = "model_data"
DATA_DIR = "../vim2/preprocessed"
RESULTS_SAVE_DIR = "../vim2/results/"

n_plot = 40
batch_size = 15
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'vim2_test')
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

n_samples = 8098
n_batches = int(np.floor(n_samples/batch_size))

#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', dim_ordering=dim_ordering)
#X_test = test_generator.create_all()
#[int(vim2_stim2.shape[0] / batch_size)

for b in (range(n_batches - 100)):
    if time.time() - starttime > 3600:
        break

    b = b+100
    X_test = np.zeros([batch_size, nt,  128, 160,3])
    for i in range(batch_size):
        X_test[i,:,:,:,:] = hkl.load(test_file +str(int(b*batch_size+i)) +".hkl")
    X_test = np.transpose(X_test, (0, 1, 4, 2, 3))
    X_test /= 255
    errs1 = test_model.predict(X_test)
    print errs1.shape
    hkl.dump(errs1[:,9,:], RESULTS_SAVE_DIR + "errors"+str(b) + ".hkl")

#X_hat = test_model.predict(X_test[1], batch_size)
#test_model._make_predict_function()
#f = test_model.predict_function
#errs1 = f(X_test[0])

# 











