import numpy as np
import os
import h5py
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma
from PIL import Image
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

class Preprocessing:
    def __init__(self, dir_path):
        self._dir_path = dir_path
        
    def readData(self):
        class_list = os.listdir(self._dir_path)
        self.images_training = []
        self.images_testing = []

        for classes in class_list:
            image_list = os.listdir(self._dir_path + classes + '/')
            success=0
            fail=0
            for image in image_list:
                try:
                    im=Image.open(self._dir_path+classes+'/'+image).convert('L').resize((256,256))
                    if classes in ['airplane', 'buildings', 'river']:
                        self.images_testing.append(np.array(im))
                    else:
                        self.images_training.append(np.array(im))
                    success+=1
                except:
                    fail+=1
            print('img class = ', classes, ' load success=%d, fail=%d' %(success, fail))

        self.images_training=np.array(self.images_training)
        self.images_testing=np.array(self.images_testing)
        
        
    def noiseImageGenerating(self, L_list, file_name):
        f = h5py.File(file_name, 'w')
        f.create_dataset("training/X", self.images_training.shape, np.uint8, chunks = True)
        f.create_dataset("testing/X", self.images_testing.shape, np.uint8, chunks = True)
        f["training/X"][...] = self.images_training
        f["testing/X"][...] = self.images_testing
                
        for L in L_list:
            n_training,x,y = self.images_training.shape
            n_testing,_,_ = self.images_testing.shape
            #scale = 1/Lambda = 1/L
            rv=gamma(a=L, scale=1./L)
            N_training = rv.rvs(size=n_training*x*y).reshape(n_training,x,y)
            N_testing = rv.rvs(size=n_testing*x*y).reshape(n_testing,x,y)
            Z_training = np.float32(self.images_training * N_training)
            Z_testing = np.float32(self.images_testing * N_testing)

            print('L=%d, noise mean and var : %f, %f' %(L, rv.stats()[0], rv.stats()[1]))

            f.create_dataset('training/L='+str(L), Z_training.shape, np.float32, chunks = True)
            f['training/L='+str(L)][...] = Z_training

            f.create_dataset('testing/L='+str(L), Z_testing.shape, np.float32, chunks = True)
            f['testing/L='+str(L)][...] = Z_testing
            
    
    def sampling(self, L_list, load_path, save_path, n_total = 1800, n_samples = 400):
        f = h5py.File(load_path, 'r')
        g = h5py.File(save_path, 'w')

        indexes = np.arange(n_total)
        np.random.shuffle(indexes)
        indexes = indexes[:n_samples]

        X=f['training/X'][...][indexes]
        _,x,y= X.shape

        g.create_dataset("training/X", (n_samples, x, y), np.uint8, chunks = True)
        g['training/X'][...] = X

        for L in L_list:
            g.create_dataset('training/L='+str(L), (n_samples, x, y), np.float32, chunks = True)
            g['training/L='+str(L)][...] = f['training/L='+str(L)][...][indexes]
            
            
            
            
            

class Data:
    def __init__(self, n_tr=400, tr_data_path = None):
        if tr_data_path != None:
            with h5py.File(tr_data_path, 'r') as f:
                self._X_train = f["training/X"][:n_tr] / 255.
        self._X_crop = np.zeros(1)
        return
    

    def _cropper(self, X, l_crop = 40, s = 10):
        n,lx,_ = X.shape
        n_crop = int((lx-l_crop)/s) + 1
        X_crop = []
        #numpy implementation is better.
        
        for i in range(n):
            for j in range(n_crop):
                for k in range(n_crop):
                    X_crop.append(X[i, s*j : s*j+l_crop , s*k : s*k+l_crop])
        print('data in: ',X.shape, 'data out:',np.array(X_crop).shape)
        return np.array(X_crop)

    
    def get_X_training(self, l_crop=None, s=None):
        if l_crop != None:
            return self._cropper(self._X_train, l_crop, s)
        return self._X_train
    
    
    def get_Z_training(self, L, l_crop=None, s=None):
        if l_crop != None:
            if self._X_crop.shape == np.zeros(1).shape:
                self._X_crop = self._cropper(self._X_train, l_crop, s)
            n, x, y = self._X_crop.shape
            rv=gamma(a=L, scale=1./L)
            N = rv.rvs(size=n*x*y).reshape(n,x,y)
            return self._X_crop * N

        n, x, y = self._X_train.shape
        rv=gamma(a=L, scale=1./L)
        N = rv.rvs(size=n*x*y).reshape(n,x,y)
        return self._X_train * N
    
    def get_blind(self):
        self._X_crop = self._cropper(self._X_train, 40, 10)
        n, x, y = self._X_crop.shape
        N = np.zeros((n,x,y))
        for i in range(n//121):
            L = 12*np.random.random()+0.5
            rv=gamma(a=L, scale=1./L)
            N[121*i:121*(i+1),:,:] = rv.rvs(size=121*x*y).reshape(121,x,y)
        return self._X_crop * N    
    

def get_test_data(te_data_path, L, n_ft):
    with h5py.File(te_data_path, 'r') as f:
        return (f["testing/X"][:n_ft] / 255., f["testing/L="+str(L)][:n_ft] / 255.)
    
def get_test_data_list(te_data_path, L_list, n_ft):
    with h5py.File(te_data_path, 'r') as f:
        return_list = [ f["testing/X"][:n_ft] / 255. ]
        for L in L_list:
            return_list.append(f["testing/L="+str(L)][:n_ft] / 255.)
        return return_list
    
def flipper(x,z):
        Z = []
        X = []
        z_flip = np.flip(z, axis=0)
        x_flip = np.flip(x, axis=0)
        for j in range(4):
            Z.append(np.rot90(z, j))
            Z.append(np.rot90(z_flip, j))
            X.append(np.rot90(x, j))
            X.append(np.rot90(x_flip, j))
            
        Z = np.array(Z).reshape(8,256,256,1)
        X = np.array(X).reshape(8,256,256,1)
        return X, Z
    
def flipper_real(z):
        Z = []
        z_flip = np.flip(z, axis=0)
        for j in range(4):
            Z.append(np.rot90(z, j))
            Z.append(np.rot90(z_flip, j))
            
        Z = np.array(Z)
        return Z

def unflipper_real(z):
    x_hat = np.zeros((z.shape[1],z.shape[2]))
    for j in range(4):
        x_hat += 1/8. * np.rot90(z[2*j], 4-j)
        x_hat += 1/8. * np.flip(np.rot90(z[2*j+1], 4-j), axis=0)
        
    return x_hat