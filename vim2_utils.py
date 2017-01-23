import h5py as h
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from matplotlib import pyplot as plt

FILE_IN = "../vim2/vim-2/Stimuli.mat"
FILE_OUT = "../vim2/vim2_data/"

image_shape = (3, 128, 160)


f = h.File("../vim2/vim-2/Stimuli.mat", "r")
#d1 = f.get("st")
d2 = f.get("sv")

#vim2_stim1 = np.array(d1)
vim2_stim2 = np.array(d2)

#print "stim1 shape"
#print vim2_stim1.shape
print "stim2 shape"
print vim2_stim2.shape


im=plt.imshow(vim2_stim2[0,:,:,:])
for index in range(vim2_stim2.shape[0]):
    pic = plt.imshow(vim2_stim2[0,:,:,:])
    im.set_data(pic)
    plt.pause(0.01)
plt.show()
"""
newim = imresize(vim2_stim2, 
                 (int(vim2_stim2.shape[0]/1.5),) + image_shape) 
print newim.shape()
                  



def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = (im.shape[1] - desired_sz[1]) / 2
    im = im[:, d:d+desired_sz[1]]
    return im
"""
