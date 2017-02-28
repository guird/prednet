import h5py as h
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from video_tools import ani_frame
from matplotlib import pyplot as plt


FILE_IN = "../vim2/vim-2/Stimuli.mat"
FILE_OUT = "../vim2/vim2_data/"

image_shape = (3, 128, 160)

end_batch_size = 10 #10 frames per batch in the final output
batch_size = 10 #that means 15 on the real deal


f = h.File("../vim2/vim-2/Stimuli.mat", "r")
#d1 = f.get("st")
d2 = f.get("sv")

#vim2_stim1 = np.array(f.get("st"), np.uint8)
vim2_stim2 = np.array(d2,np.uint8)

#stimuli are in shape (frames, 3, 128,128), go figure
#vim2_stim1 = np.transpose(vim2_stim1, [0,3,2,1])
vim2_stim2 = np.transpose(vim2_stim2, [0,3,2,1])



#print "stim1 shape"
#print vim2_stim1.shape

                  


#Process_vid changes the dimensions of a video of shape(frames, x,y,3) 

def process_vid(vid, desired_shape):
    #scale video dims
    newvid = np.zeros((vid.shape[0],) + (desired_shape))

    for imindex in range(vid.shape[0]):
        newim = imresize(vid[imindex, :,:,:], desired_shape)
        newvid[imindex, :,:,:] = newim
    
    #scale video frames to reflect fps
         

 


    return newvid

#ani_frame(vim2_stim2, 15, "stim2_raw")
#ani_frame(vim2_stim1[0:300], 15, "stim1_raw")
#vim2_stim2 = process_vid(vim2_stim2, (128,160,3), 1/1.5)




#vim2_stim2 = np.uint8(vim2_stim2)


for i in range(vim2_stim2.shape[0]-batch_size):
    vim2_stim2p = process_vid(vim2_stim2[i:i + batch_size], (128,160,3))
    print vim2_stim2p.shape
    hkl.dump(vim2_stim2p, "../vim2/preprocessed/vim2_test"+str(i)+".hkl")

"""for i in range(int(vim2_stim2.shape[0]/batch_size)):
    hkl.dump(vim2_stim2[i*batch_size:(i+1)*batch_size], "../vim2/preprocessed/im2_test+"+str(i)+".hkl")
"""
#ani_frame(vim2_stim2, 10, "stim2_preprocessed")




