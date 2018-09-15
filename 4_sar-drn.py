from Module.data import Data
from Module.models import Sar_drn
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import numpy as np
import math
import h5py



#hyperparameter
batch_size = 128
epochs = 50
lr = 1e-2
n_tr = 400
n_te = 300
L = 1.0
std = np.sqrt(1/L)

#path
experiment_name = 'sardrn_L'+str(L)
tr_data_path = './Data/Images_sampled.h5'
te_data_path = './Data/Images.h5'
weight_path = './Weight/' + experiment_name
result_path = './Result/' + experiment_name


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate




#load data
#x_tr is in [0,1], z_tr = x_tr*n
data = Data(n_tr=n_tr, tr_data_path=tr_data_path)
x_tr = data.get_X_training(l_crop=40, s=10)


#load model
model = Sar_drn(input_shape=(40,40,1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
model.summary()


#callback
lrate = LearningRateScheduler(step_decay)
tensorboard = TensorBoard(log_dir="Log/"+experiment_name, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
callback_list = [tensorboard, lrate]


#train
for epoch in range(epochs):
    z_tr = data.get_Z_training(L=L, l_crop=40, s=10)
    model.fit(z_tr.reshape(-1,40,40,1), (z_tr-x_tr).reshape(-1,40,40,1), batch_size=batch_size, epochs=epoch+1, callbacks=callback_list, initial_epoch = epoch)
model.save_weights(weight_path)
    
    

    
    
    
#evaluate
with h5py.File(result_path, 'w') as f:
    f.create_dataset("SSIM", (n_te,), np.float32, chunks = True)
    f.create_dataset("PSNR", (n_te,), np.float32, chunks = True)

with h5py.File(te_data_path, 'r') as f:
    x_te, z_te = f["testing/X"][:n_te] / 255., f["testing/L="+str(L)][:n_te] / 255.

del model
model = Sar_drn(input_shape=(256,256,1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
model.load_weights(weight_path)


N_hat = (model.predict(z_te.reshape(-1,256,256,1), verbose=0)).reshape(-1,256,256)

X_hat = z_te - N_hat
for i in range(n_te):
    with h5py.File(result_path) as f:
        f['SSIM'][i] = compare_ssim(np.float32(x_te[i]), X_hat[i])
        f['PSNR'][i] = compare_psnr(np.float32(x_te[i]), X_hat[i], 1)
