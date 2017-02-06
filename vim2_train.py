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

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
#from data_utils import SequenceGenerator 

WEIGHTS_DIR = "model_data"
DATA_DIR = "../vim2/preprocessed"
RESULTS_SAVE_DIR = "../vim2/results"
WEIGHTS_OUT_DIR = "vim2_weights"

n_plot = 40
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
training_file = os.path.join(DATA_DIR, 'vim2_train')
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

num_epochs = 5
num_batches = 5#let's just say for now


# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'error'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
errors = test_prednet(inputs)
model = Model(input=inputs, output=errors)
model.compile(loss='mean_absolute_error', optimizer = 'adam')

#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', dim_ordering=dim_ordering)
#X_test = test_generator.create_all()
#[int(vim2_stim2.shape[0] / batch_size)
X_train = np.zeros([539, batch_size, 128, 160,3])


for i in (range(539)):
    X_train[i,:,:,:,:] = hkl.load(training_file + str(i) +".hkl")
X_train = np.transpose(X_train, (0, 1, 4, 2, 3))

errors_shape = errors.shape

for e in range(num_epochs):
    for i in np.random.shuffle(range(num_batches)):
        model.train_on_batch(X_train[i,:,:,:,:], np.zeros(errors_shape))
                                   


#X_hat = np.transpose(X_hat, (0, 1, 4, 2, 3))

