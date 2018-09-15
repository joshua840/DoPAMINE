from keras import backend as K


def sl_loss(y_true,y_pred):
    A = y_pred[:,:,:,0]
    B = y_pred[:,:,:,1]
    X = y_true[:,:,:,0]
    Z = y_true[:,:,:,1]
    return K.mean(K.square(X - A*Z - B))/2.
        