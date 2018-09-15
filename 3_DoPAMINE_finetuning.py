from Module.customLoss import sl_loss
from Module.models import Dopamine
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler
#FT
from Module.data import Data, get_test_data, flipper
from keras import backend as K
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import h5py

import numpy as np
import math



#hyperparameter
epochs_ft = 10
lr = 1e-3
lr_ft = 1e-4
n_tr = 400
n_ft = 300
L = 1.0
std = np.float32(np.sqrt(1/L))


#path
experiment_name = 'DoPAMINE_L'+str(L)
te_data_path = './Data/Images.h5'
weight_path = './Weight/' + experiment_name
ft_result_path = './Result/' + experiment_name+'_FlipAvgFT'


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def ft_loss(y_true,y_pred):
    A=y_pred[0,:,:,0]
    B=y_pred[0,:,:,1] 
    Z=y_true[0,:,:,0]
    return K.mean(K.square((1-A)*Z-B) + K.square(std*Z)*(2*A-1)/(1+K.square(std)))/2.





#load data
x_te, z_te = get_test_data(te_data_path, L, n_ft)

#save_file 
with h5py.File(ft_result_path, 'w') as fi:            
    fi.create_dataset("SSIM", (n_ft,), np.float32, chunks = True)
    fi.create_dataset("PSNR", (n_ft,), np.float32, chunks = True)
        
        
model = Dopamine((256,256,1))
    
def run(i):
#load model
    model.compile(loss=ft_loss, optimizer=Adam(lr=lr_ft))

    #fit
    model.load_weights(weight_path)
    
    x8, z8 = flipper(x_te[i], z_te[i])
    
    for epoch in range(epochs_ft):
        hist = model.fit(z8, z8, verbose=0, batch_size=1, epochs=epoch+1, initial_epoch = epoch)
                     
    #evaluate
    AB = np.array(model.predict(z8, batch_size=8, verbose=0))
    x_hat_arr = AB[:,:,:,0] * z8[:,:,:,0] + AB[:,:,:,1]
    x_hat = np.zeros((256,256))
    for j in range(4):
        x_hat += 1/8. * np.rot90(x_hat_arr[2*j], 4-j)
        x_hat += 1/8. * np.flip(np.rot90(x_hat_arr[2*j+1], 4-j), axis=0)

    psnr = compare_psnr(np.float32(x_te[i]), x_hat, 1)
    ssim = compare_ssim(np.float32(x_te[i]), x_hat)        
    with h5py.File(ft_result_path) as fi:
        fi['PSNR'][i] = psnr
        fi['SSIM'][i] = ssim
    print(i,'th image finished, psnr:', psnr)
    
for i in range(n_ft):
    run(i)