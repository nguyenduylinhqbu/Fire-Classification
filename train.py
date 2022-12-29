# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

#Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import layers
import keras

#confusion_matrix
from sklearn.metrics import confusion_matrix

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
import os
from keras.losses import categorical_crossentropy
#from resnet50 import resnet50
from cnn_multi import Lenet, MobileNet_MODEL, VGG19_MODEL, VGG13_LRELU_2, VGG16_MODEL, InceptionV3_MODEL, DenseNet121_MODEL, NASNetMobile_MODEL, SqueezeNet, combine_4SC_2_1_V3, MobileNetV2_MODEL, Xception_MODEL


#--Edit
from imutils import paths
import random
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import np_utils
import seaborn as sns

print(os.listdir("./Dataset/FireNet"))
#name_file = "Lenet"
#name_file = "VGG16"
#name_file = "MobileNetV1"
#name_file = "VGG19"
#name_file = "Xception"
#name_file = "VGG13"
#name_file = "InceptionV3"
#name_file = "DenseNet121"
#name_file = "NASNetMobile"
#name_file = "MobileNetV2"
#name_file = "SqueezeNet"
name_file = "ProposedNetwork"



print('----------' + name_file)
def getLabel(id):
    return ['NoFire', 'Fire'][id]
#----- dataloader
print("[INFO] loading images...")
data = []
labels = []
path = './Dataset/FireNet/'
norm_size = 224
#norm_size = 100
num_classes = 2
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(path)))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (norm_size, norm_size))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = int(imagePath.split(os.path.sep)[-2])       
    labels.append(label)  
        
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.3, random_state=42)  
y_train = to_categorical(y_train, num_classes=num_classes)
y_true = y_test
y_test = to_categorical(y_test, num_classes=num_classes) 


#------ End of dataloader



print("total Train:", len(y_train))
#print("total Valid:", len(y_valid))
print("total Test:", len(y_test))

print('----------')

input_shape=(224,224,3)
#model = Lenet(input_shape, num_classes) #7
#model = VGG16_MODEL(input_shape, num_classes) #10
#model = MobileNet_MODEL(input_shape, num_classes) #11
#model = VGG19_MODEL(input_shape, num_classes) #12
#model = Xception_MODEL(input_shape, num_classes) #13
#model = VGG13_LRELU_2(input_shape, num_classes) #16
#model = InceptionV3_MODEL(input_shape, num_classes) #17
#model = DenseNet121_MODEL(input_shape, num_classes) #18
#model = NASNetMobile_MODEL(input_shape, num_classes) #19
#model = MobileNetV2_MODEL(input_shape, num_classes) #31
#model = SqueezeNet(input_shape, num_classes) #33
model = combine_4SC_2_1_V3(input_shape, num_classes) #37



model.compile(loss=categorical_crossentropy,
             optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['acc'])


print('----------')
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
model.layers[0].trainable

from keras import callbacks
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stop = EarlyStopping('val_acc', patience=500)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.75, patience=10, verbose=1)

filepath="./Model/" + name_file + ".hdf5"
model_checkpoint = ModelCheckpoint(filepath, 'val_acc', verbose=1,save_best_only=True)

callbacks_list = [model_checkpoint, csv_log, early_stop, reduce_lr]

hist = model.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)

model.save( "./Model/" + name_file +"_LAST.hdf5")

# visualizing losses and accuracy
from IPython import get_ipython

#Plot Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
rcParams['figure.figsize'] = 10, 8
#results = model.predict_classes(X_test)
y_prob = model.predict(X_test) 
results = y_prob.argmax(axis=-1)

cm = confusion_matrix(np.where(y_test == 1)[1], results)
labels = ['NoFire', 'Fire']

title='Confusion matrix'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=10,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

fig = plt.figure()
plot_confusion_matrix(cm, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
fig.savefig('./Plot/'+name_file+'_ConfusionMatrix.png')
# -----

#Plot Loss and ACC
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='Train_loss')
plt.plot(epochs,val_loss,'b', label='Val_loss')
plt.title('Train_loss vs Val_loss')
plt.legend()
plt.savefig("./Plot/" +name_file+'_Loss.png',dpi=300)
#plt.figure()
plt.show()


plt.plot(epochs,train_acc,'r', label='Train_acc')
plt.plot(epochs,val_acc,'b', label='Val_acc')
plt.title('Train_acc vs Val_acc')
plt.legend()
plt.savefig("./Plot/" +name_file+'_Acc.png',dpi=300)
#plt.figure()
plt.show()
# -------
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



model.compile(loss=categorical_crossentropy,
             optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['acc', f1_m, precision_m, recall_m])




loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', loss)
print('---- METRIC----')
print('Test accuracy:', accuracy)
print('F1 score:', f1_score)
print('Precision:', precision)
print('Recall:', recall)
print('---- END METRIC----')
#-------Plot Best model result

from keras.models import load_model
model_path = "./Model/" + name_file + ".hdf5"
model = load_model(model_path, compile=False)
res = model.predict(X_test[0:18]).argmax(axis=-1)
#res = model.predict_classes(X_test[0:18])
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    b,g,r =cv2.split(X_test[i])
    X_test[i]=cv2.merge([r,g,b])

    plt.imshow(X_test[i])
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=10)
# show the plot
plt.savefig('./Plot/'+name_file+'_Best.png')
plt.show()

# ------ Plot Last Model
from keras.models import load_model
model_path =  "./Model/" + name_file +"_LAST.hdf5"
model = load_model(model_path, compile=False)


res = model.predict(X_test[0:18]).argmax(axis=-1)
#res = model.predict_classes(X_test[0:18])
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    b,g,r =cv2.split(X_test[i])
    X_test[i]=cv2.merge([r,g,b])

    plt.imshow(X_test[i])
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=10)
# show the plot
plt.savefig('./Plot/'+name_file+'_Last.png')
plt.show()
