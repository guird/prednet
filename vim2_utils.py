import h5py as h
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from video_tools import ani_frame
from matplotlib import pyplot as plt


FILE_IN = "../vim2/vim-2/Stimuli.mat"
FILE_OUT = "../vim2/vim2_data/"

image_shape = (3, 128, 160)


f = h.File("../vim2/vim-2/Stimuli.mat", "r")
d1 = f.get("st")
d2 = f.get("sv")

vim2_stim1 = np.array(d1)
vim2_stim2 = np.array(d2)

#stimuli are in shape (frames, 3, 128,128), go figure
vim2_stim1 = np.transpose(vim2_stim1, [0,3,2,1])
vim2_stim2 = np.transpose(vim2_stim2, [0,3,2,1])



print "stim1 shape"
print vim2_stim1.shape
print "stim2 shape"
print vim2_stim2.shape


                  


#Process_vid changes the dimensions of a video of shape(frames, x,y,3) 

def process_vid(vid, desired_shape, fps_ratio):
    #scale video dims
    newvid = np.zeros((vid.shape[0],) + (desired_shape))

    for imindex in range(vid.shape[0]):
        newim = imresize(vid[imindex, :,:,:], desired_shape)
        newvid[imindex, :,:,:] = newim
    
    #scale video frames to reflect fps
         
    finalvid = np.zeros((int(newvid.shape[0]*fps_ratio), 
                         newvid.shape[1],
                         newvid.shape[2],
                         newvid.shape[3]))
    
    for cross_sec in range(newvid.shape[2]):
        newim = imresize(newvid[:,:,cross_sec,:],
                          (int(newvid.shape[0]*fps_ratio),)+
                          (newvid.shape[1],newvid.shape[3]))
        finalvid[:,:,cross_sec,:] = newim
 


    return finalvid

ani_frame(vim2_stim2, 15, "stim2_raw")
ani_frame(vim2_stim1, 15, "stim1_raw")
vim2_stim2 = process_vid(vim2_stim2, (128,160,3), 1/1.5)
vim2_stim1 = process_vid(vim2_stim1, (128,160,3), 1/1.5)

ani_frame(vim2_stim2, 10, "stim2_preprocessed")
ani_frame(vim2_stim1, 10, "stim1_preprocessed")
