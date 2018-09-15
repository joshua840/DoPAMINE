from keras import backend as K
from keras.models import Model, Input
from keras.layers import Conv2D, ZeroPadding2D, Add, Average, Lambda, ReLU, Multiply
import numpy as np
import tensorflow as tf
import keras


def Sar_drn(input_shape, filter_size = 64):
    f = filter_size
    
    x = Input(shape=input_shape)
    L1 = Conv2D(64, (3,3), activation='relu', dilation_rate=1, padding='same', kernel_initializer = 'he_normal')(x)
    L2 = Conv2D(64, (3,3), activation='relu', dilation_rate=2, padding='same', kernel_initializer = 'he_normal')(L1)
    L3 = Conv2D(64, (3,3), activation='relu', dilation_rate=3, padding='same', kernel_initializer = 'he_normal')(L2)
    A1 = Add()([L1, L3])
    L4 = Conv2D(64, (3,3), activation='relu', dilation_rate=4, padding='same', kernel_initializer = 'he_normal')(A1)
    L5 = Conv2D(64, (3,3), activation='relu', dilation_rate=3, padding='same', kernel_initializer = 'he_normal')(L4)
    L6 = Conv2D(64, (3,3), activation='relu', dilation_rate=2, padding='same', kernel_initializer = 'he_normal')(L5)
    A2 = Add()([L4, L6])
    L7 = Conv2D(1, (3,3), activation='linear', dilation_rate=1, padding='same', kernel_initializer = 'he_normal')(A2)

    return Model(inputs=x, outputs=L7)




def ResNet20_2(input_shape,
             batch_norm = False,
             multiplier = 1.,
             f = 64,
             kernel_initializer = 'he_normal',
             bias_initializer = 'zeros'):
    
    x_in = Input(shape=input_shape)
    x = Conv2D(f, (3,3), activation='relu', padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(x_in)

    for i in range(20):
        if batch_norm == True:
            x = BatchNormalization()(x)
        a = ReLU()(x)
        a = Conv2D(f, (3,3), padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(a)
        if batch_norm == True:
            a = BatchNormalization()(a)
        a = ReLU()(a)
        a = Conv2D(f, (3,3), padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(a)
        x = Add()([x,a])
        x = Lambda(lambda x: x * multiplier)(x)
    x = ReLU()(x)        
    res = Conv2D(1, (3,3), padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(x)

    model = Model(inputs=x_in, outputs=res)

    return model





def Dopamine(input_shape,
             f = 64, 
             num_layer = 21,
             num_resnet_block = 5,
             filter_size = 3,
             multiplier_add = np.sqrt(1/2.),
             multiplier_combine = np.sqrt(1/21.),
             kernel_initializer = 'he_normal',
             bias_initializer = 'zeros'):
    
    def maskedConv(udlr, x, i, name, mode):
        u, d, l, r = udlr
        x_shape = K.int_shape(x)
        if name == 'H':
            convname = mode+'_'+name+'_mask_conv_'+str(i)
        else:
            convname = mode+'_V_'+str(i)

        with tf.name_scope(mode+'_'+name+str(i)):
            if  i==0 and u+d+l+r==0:    
                x = Conv2D(f, (1+u+d, 1+l+r), padding='valid',
                         name=convname, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(x)
                

                if mode == 'LU':
                    x = ZeroPadding2D(padding=((0,0),(1,0)), name=mode+'_H_padding')(x)
                    res = Lambda(lambda x: x[:,:,:x_shape[2],:], name=mode+'_H_shifting')(x)
                else:
                    x = ZeroPadding2D(padding=((0,0),(0,1)), name=mode+'_H_padding')(x)
                    res = Lambda(lambda x: x[:,:,1:,:], name=mode+'_H_shifting')(x)                
            else:
                if i!=0:
                    x = ReLU(name='relu_'+convname)(x)
                res = ZeroPadding2D(padding=((u,d),(l,r)), name=mode+'_'+name+'_mask_pad_'+str(i))(x)
                res = Conv2D(f, (1+u+d, 1+l+r), padding='valid', 
                         name=convname, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(res)
                if i == 0 and u+d+l+r==2:
                    if mode == 'LU':
                        x = ZeroPadding2D(padding=((1,0),(0,0)), name=mode+'_V_padding')(res)
                        res = Lambda(lambda x: x[:,:x_shape[2],:,:], name=mode+'_V_shifting')(x)
                    else:
                        x = ZeroPadding2D(padding=((0,1),(0,0)), name=mode+'_V_padding')(res)
                        res = Lambda(lambda x: x[:,1:,:,:], name=mode+'_V_shifting')(x)    
                        
                    
        return res

    
    def passLayer(mode, V, H, i):
        if mode == 'LU':
            udlr_v = (filter_size//2, 0, filter_size//2, filter_size//2)
            udlr_h = (0,0,filter_size//2,0)
        else:
            udlr_v = (0, filter_size//2, filter_size//2, filter_size//2)
            udlr_h = (0,0,0,filter_size//2)
        if i == 0:
            udlr_h = (0,0,0,0)
            udlr_v = (0, 0, filter_size//2, filter_size//2)
            
        V_masked = maskedConv(udlr_v, V, i, name='V', mode=mode)
        with tf.name_scope(mode+'_V_feed'+str(i)):
            x = ReLU(name=mode+'_V_feed_relu_'+str(i))(V_masked)
            V_feedMap = Conv2D(f, (1, 1), name=mode+'_V_feed_conv_'+str(i), bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(x)
        H_masked = maskedConv(udlr_h, H, i, name='H', mode=mode)


        with tf.name_scope(mode+'_HaddV'+str(i)):
            H_out = Add(name=mode+'_addHV_'+str(i))([H_masked, V_feedMap])
            H_out = Lambda(lambda x: x * multiplier_add)(H_out)
            H_out = ReLU(name=mode+'relu_addHV_'+str(i))(H_out)
            H_out = Conv2D(f, (1,1), name=mode+'_H_1x1_'+str(i), bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(H_out)
            if i != 0:
                H_out = Add(name=mode+'_H_'+str(i))([H_out, H])
                H_out = Lambda(lambda x: x * multiplier_add)(H_out)
        return V_masked, H_out


    def combLayer(LU, RD, i):
        c = Add(name='M_'+str(i))([LU, RD])
        c = Lambda(lambda x: x * multiplier_add)(c)
        c = ReLU()(c)
        return Conv2D(f, (1, 1), name='Mconv_'+str(i), bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(c)
        



    def make_model(input_shape):
        x = Input(shape=input_shape)
        M = []

        with tf.name_scope('LUpassLayer0'):
            VLU, HLU = passLayer('LU', x, x, 0)
        with tf.name_scope('RDpassLayer0'):
            VRD, HRD = passLayer('RD', x, x, 0)

        M.append(combLayer(HLU, HRD, 0))
        
        for i in range(1,num_layer):        
            with tf.name_scope('LUpassLayer'+str(i)):
                VLU, HLU = passLayer('LU', VLU, HLU, i)
            with tf.name_scope('RDpassLayer'+str(i)):
                VRD, HRD = passLayer('RD', VRD, HRD, i)
                
            M.append(combLayer(HLU, HRD, i))

        M_add = Add()(M)
        M_add = Lambda(lambda x: x * multiplier_combine)(M_add)

        for i in range(num_resnet_block):
            M_add1 = ReLU()(M_add)
            M_add1 = Conv2D(f, (1, 1), padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(M_add1)
            M_add1 = ReLU()(M_add1)
            M_add1 = Conv2D(f, (1, 1), padding='same', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(M_add1)
            M_add = Add()([M_add,M_add1])
            M_add = Lambda(lambda x: x * multiplier_add)(M_add)
            
            
        M_add = ReLU(name = '1x1relu'+str(i+1))(M_add)
        res = Conv2D(2, (1, 1), activation='linear', padding='valid', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer)(M_add)

        model = Model(inputs=x, outputs=res)

        return model
    return make_model(input_shape)