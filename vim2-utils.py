import h5py as h
import numpy as np

f = h.File("../vim2/vim-2/Stimuli.mat", "r")
d1 = f.get("st")
d2 = f.get("sv")

vim2_stim1 = np.array(d1)
vim2_stim2 = np.array(d2)
