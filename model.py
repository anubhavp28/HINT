from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow
from zweeBasic import *


import PIL
#import keras_metrics as km
import keras.backend as K

from warnings import filterwarnings
filterwarnings('ignore')

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=hyperparameter((64,64,3),"input_shape"),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy', precision, recall, fmeasure, 'binary_crossentropy', \
    'mean_squared_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', f1])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
train_set = train_datagen.flow_from_directory('train',
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set
test_set = test_datagen.flow_from_directory('test',
                                           target_size=(64,64),
                                           batch_size = 32,
                                           class_mode='binary',
                      
                                           shuffle=False)
#Test Set /no output available
#test_set1 = test_datagen.flow_from_directory('test1',
                                            #target_size=(64,64),
                                           # batch_size=32,
                                            #shuffle=False)

def train():
    return classifier.fit_generator(train_set,
                        steps_per_epoch=hyperparameter(800,"steps_per_epoch"), 
                        epochs = hyperparameter(20, "epochs"),
                        validation_data = test_set,
                        validation_steps = hyperparameter(20, "validation_steps"), 
                        );


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

@predict_func
def predict(filename):
    img1 = image.load_img(filename, target_size=(64, 64))
    img = image.img_to_array(img1)
    img = img/255
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
    if(prediction[:,:]>0.5):
        predicted_label = 'dog'
    else:
        predicted_label = 'cat'
    return predicted_label


