from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense, Concatenate, Add, PReLU, LeakyReLU, concatenate
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras import regularizers
from keras.activations import linear
from keras.layers import multiply


from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import MobileNet, MobileNetV2
from keras.applications.xception import Xception
from keras.applications import InceptionV3
from keras.applications import DenseNet121
from keras.applications import NASNetMobile

import tensorflow as tf

import numpy as np
import cv2

from Attention import cbam_block, se_block
#from Inception import inception_module

#Proposed Network 
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)#Separable
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)#Separable
    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)#Separable
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)#Separable
    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)#Separable
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)#Separable
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
    
def combine_4SC_2_1_V3(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # Block1-16 channels
    img_input = Input(input_shape)
    x = Conv2D(16, (7, 7), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    #residual0 = Conv2D(10, (1, 1), strides=(16, 16),
                      #padding='same', use_bias=False)(x)
    #residual0 = BatchNormalization()(residual0)


    x = Conv2D(16, (7, 7), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    #x = Dropout(0.5)(x)

    # Block2-32 channels
    #residual = Conv2D(16, (3, 3), strides=(2, 2), use_bias=False)(x)
    #residual = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(residual)
    #residual = Conv2D(32, (1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
    #residual = BatchNormalization()(residual)

    x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # 1st inception
    #x = inception_module(x, 64, 96, 128, 16, 32, 32)
    #x = inception_module(x, 32, 24, 32, 24, 32, 32)
    
    #Att 1
    #x = cbam_block(x)
    x = se_block(x)
    #x = BAM(x, batch_norm_params)

    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    x = SeparableConv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    #x = Dropout(0.5)(x)
    #x = layers.add([x, residual])

    # module 3-64 channels
    residual = SeparableConv2D(32, (3, 3), strides=(1, 1),padding='same', use_bias=False)(x)
    residual = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(residual)
    residual = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
    residual = BatchNormalization()(residual)
    
    # 2nd inception
    #x = inception_module(x, 128, 128, 128, 32, 24, 32)
    #x = inception_module(x, 64, 48, 64, 48, 64, 64)
    x = inception_module(x, 32, 24, 32, 24, 32, 32)
    
    
    #Att 2
    #x = cbam_block(x)
    x = se_block(x)
    #x = BAM(x, batch_norm_params)
    
    x = Conv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    #x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            #use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    #x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            #use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    #x = layers.add([x, residual])

    # module 4 - 128 channels
    #residual = Conv2D(64, (3, 3), strides=(1, 1),padding='same', use_bias=False)(x)
    #residual = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(residual)
    #residual = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
    #residual = BatchNormalization()(residual)
    
    # 3rd inception
    #x = inception_module(x, 64, 96, 128, 16, 32, 32)
    #x = inception_module(x, 32, 24, 32, 24, 32, 32)
    x = inception_module(x, 32, 24, 32, 24, 32, 32)
    
    #Att 3
    #x = cbam_block(x)
    x = se_block(x)
    #x = BAM(x)
    
    #Connection 1
    x = layers.add([x, residual])
    
    
    x = Conv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    #x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            #use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    #x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            #use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    #x = layers.add([x, residual])

    # Final block
    residual = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    residual = SeparableConv2D(2, (1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
    residual = BatchNormalization()(residual)
    
    # 4th inception
    #x = inception_module(x, 128, 128, 192, 32, 96, 64)
    #x = inception_module(x, 64, 48, 64, 48, 64, 64)
    x = inception_module(x, 32, 24, 32, 24, 32, 32)
    
    #Att 4
    #x = cbam_block(x)
    x = se_block(x)
    #x = BAM(x)
    
    x = SeparableConv2D(256, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    #x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            #use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # 5th inception
    x = inception_module(x, 32, 24, 32, 24, 32, 32)
    #Att 5
    #x = cbam_block(x)
    x = se_block(x)
    #x = BAM(x)
    
    x = SeparableConv2D(num_classes, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    
    #Add 2
    x = layers.add([x, residual])

    #x = SeparableConv2D(128, (3, 3), padding='same',
                        #kernel_regularizer=regularization,
                        #use_bias=False)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = SeparableConv2D(128, (3, 3), padding='same',
                        #kernel_regularizer=regularization,
                        #use_bias=False)(x)
    #x = BatchNormalization()(x)

    #x = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    #x = layers.add([x, residual])

    #x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            #padding='same')(x)
    #x = cbam_block(x)
    #x = se_block(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model


#SqueezeNet
def fire_module(input_fire, s1, e1, e3, weight_decay_l2, fireID):  
    '''
    A wrapper to build fire module
    
    # Arguments
        input_fire: input activations
        s1: number of filters for squeeze step
        e1: number of filters for 1x1 expansion step
        e3: number of filters for 3x3 expansion step
        weight_decay_l2: weight decay for conv layers
        fireID: ID for the module
    
    # Return
        Output activations
    '''
    
    # Squezee step
    output_squeeze = Convolution2D(
        s1, (1, 1), activation='relu', 
        kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding='same', name='fire' + str(fireID) + '_squeeze',
        data_format="channels_last")(input_fire)
    # Expansion steps
    output_expand1 = Convolution2D(
        e1, (1, 1), activation='relu', 
        kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding='same', name='fire' + str(fireID) + '_expand1',
        data_format="channels_last")(output_squeeze)
    output_expand2 = Convolution2D(
        e3, (3, 3), activation='relu',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding='same', name='fire' + str(fireID) + '_expand2',
        data_format="channels_last")(output_squeeze)
    # Merge expanded activations
    output_fire = Concatenate(axis=3)([output_expand1, output_expand2])
    return output_fire

def SqueezeNet(input_shape, num_classes, weight_decay_l2=0.0001):
    '''
    A wrapper to build SqueezeNet Model
    
    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions
    
    # Return
        A SqueezeNet Keras Model
    '''
    input_img = Input(shape=input_shape)
    
    conv1 = Convolution2D(
        32, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    
    maxpool1 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    
    fire2 = fire_module(maxpool1, 8, 16, 16, weight_decay_l2, 2)    
    fire3 = fire_module(fire2, 8, 16, 16, weight_decay_l2, 3)
    fire4 = fire_module(fire3, 16, 32, 32, weight_decay_l2, 4)
    
    maxpool4 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(fire4)
    
    fire5 = fire_module(maxpool4, 16, 32, 32, weight_decay_l2, 5)
    fire6 = fire_module(fire5, 32, 64, 64, weight_decay_l2, 6)
    fire7 = fire_module(fire6, 32, 64, 64, weight_decay_l2, 7)
    fire8 = fire_module(fire7, 64, 128, 128, weight_decay_l2, 8)
    
    maxpool8 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool8',
        data_format="channels_last")(fire8)
    
    fire9 = fire_module(maxpool8, 64, 128, 128, weight_decay_l2, 9)
    fire9_dropout = Dropout(0.5, name='fire9_dropout')(fire9)
    
    conv10 = Convolution2D(
        num_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)
    
    return Model(inputs=input_img, outputs=softmax)


# MobileNet V2
def MobileNetV2_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_MobileNet_1 = MobileNetV2(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_MobileNet_1.layers:
        layer.trainable = False
        
    x = model_MobileNet_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_MobileNet_1.input, output = predictions)
    
    return model

# NASNetMobile
def NASNetMobile_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_NASNetMobile_1 = NASNetMobile(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_NASNetMobile_1.layers:
        layer.trainable = False
        
    x = model_NASNetMobile_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_NASNetMobile_1.input, output = predictions)
    
    return model
# DenseNet121
def DenseNet121_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_DenseNet121_1 = DenseNet121(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_DenseNet121_1.layers:
        layer.trainable = False
        
    x = model_DenseNet121_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_DenseNet121_1.input, output = predictions)
    
    return model
# InceptionV3
def InceptionV3_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_InceptionV3_1 = InceptionV3(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_InceptionV3_1.layers:
        layer.trainable = False
        
    x = model_InceptionV3_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_InceptionV3_1.input, output = predictions)
    
    return model


InceptionV3
# Model 16   
def VGG13_LRELU_2(input_shape, num_classes):
    #model = Sequential()
    
    img_input = Input(input_shape)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    # layer2 48*48*64
    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    

    # layer3 24*24*64
    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    # layer4 24*24*128
    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    

    # layer5 12*12*128
    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    # layer6 12*12*256
    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    # layer7 12*12*256
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    

    # layer8 6*6*256
    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    # layer9 6*6*512
    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    # layer10 6*6*512
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    
    # layer11 3*3*512
    #x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    #x = LeakyReLU(alpha=0.01)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.4)(x)
    # layer12 3*3*512
    #x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    #x = LeakyReLU(alpha=0.01)(x)
    #x = BatchNormalization()(x)
    # layer13 3*3*512
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.4)(x)


    # layer14 1*1*512
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)  
    
    #x = Dropout(0.4)(x)
    #x = Dense(128)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(img_input, x)
    return model

# Xception
def Xception_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_Xception_1 = VGG16(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_Xception_1.layers:
        layer.trainable = False
        
    x = model_Xception_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_Xception_1.input, output = predictions)
    
    return model

# VGG19
def VGG19_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_vgg19_1 = VGG19(weights="imagenet", include_top=False)
    
    # Freeze all layers of the pre-trained model
    for layer in model_vgg19_1.layers:
    	layer.trainable = False
    	
    
        
    x = model_vgg19_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_vgg19_1.input, output = predictions)
    
    return model
# MobileNet
def MobileNet_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_MobileNet_1 = MobileNet(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_MobileNet_1.layers:
        layer.trainable = False
        
    x = model_MobileNet_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_MobileNet_1.input, output = predictions)
    
    return model
# VGG16
def VGG16_MODEL(input_shape, num_classes):
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    model_vgg16_1 = VGG16(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in model_vgg16_1.layers:
        layer.trainable = False
        
    x = model_vgg16_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(input = model_vgg16_1.input, output = predictions)
    
    return model

# Lenet
def Lenet(input_shape, num_classes):
    # initialize the model
    model = Sequential()
    #Input(input_shape)

    # if we are using "channel_first", update the input shape
    #if K.image_data_format() == "channel_first":
    #    inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding = "same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classififer
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model



