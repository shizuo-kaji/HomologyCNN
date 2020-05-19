import numpy as np
import keras
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, labels, batch_size=32, shuffle=True, random=True):
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.X = X
        self.random = random
        self.random_rot = False
        self.num_ch = X.shape[-1]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        # indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        if self.random_rot: # random rotation (TODO)
            rot = R.random(len(indices)).as_matrix()
            X_temp = []
            for i, k in enumerate(indices):
                X = []
                for j in range(self.num_ch):
                    X.append(ndimage.affine_transform(self.X.transpose(0,4,1,2,3)[k,j],rot[i]))
                X_temp.append(X)
            X_out = np.array(X_temp).transpose(0,2,3,4,1)
        elif self.random: # random flip and transpose
            X_temp = []
            for i, k in enumerate(indices):
                L=[0,1,2]
                random.shuffle(L)
                X = self.X[k].transpose(*L,3)
                # p = random.random()
                # if p < 0.25:
                #      X = X[::-1,::-1,:,:]
                # elif p < 0.5:
                #      X = X[:,::-1,::-1,:]
                # elif p < 0.75:
                #      X = X[::-1,:,::-1,:]
                X_temp.append(X)
            X_out = np.array(X_temp)
        else:
            X_out = self.X[indices]
#        print(X_out.shape)
        return X_out, self.labels[indices]

    def on_epoch_end(self):
        self.indices = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
