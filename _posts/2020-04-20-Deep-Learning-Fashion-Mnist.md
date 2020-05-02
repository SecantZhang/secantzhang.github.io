---
title: 'Achieving 95.42% Accuracy on Fashion-Mnist Dataset Using Transfer Learning and Data Augmentation with Keras'
date: 2020-04-20
description: 
featured_image: '/images/blogs/04-20-20-Deep-Learning-Fashion-Mnist/dl_banner.png'
---

I have most of the working code below, and I'm still updating it. 

## Background


## Google Colab Implementation

### Environment Set-up

```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import multiprocessing
from statistics import mean

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.backend import repeat_elements, expand_dims, resize_images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from scipy.stats import reciprocal

!pip install keras-tuner
from kerastuner.tuners import Hyperband
```

Sanity Check

```python
print(tf.__version__)
print(keras.__version__)
if 'COLAB_TPU_ADDR' in os.environ: 
    print('Connected to TPU') 
elif tf.test.gpu_device_name() is not '': 
    print('Connected to GPU ' + tf.test.gpu_device_name())
else:
    print('Neither connected to a TPU nor a GPU')

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
else:
    print(gpu_info)
```

Mount the google drive to the colab space: 

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### Data Preparation

```python
os.chdir("<your working directory>")
train_original = pd.read_csv("data/fashion-mnist_train.csv")
blind_test_original = pd.read_csv("data/fashion-mnist_test.csv")

X, y = train_original.iloc[:, 1:], train_original.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7980)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y = to_categorical(y)

# Convert flatted data to the image format. 
X_train_tfd = X_train.values.reshape(X_train.shape[0], 28, 28)
X_test_tfd = X_test.values.reshape(X_test.shape[0], 28, 28)
blind_test_tfd = blind_test_original.values.reshape(blind_test_original.shape[0], 28, 28)
X_tfd = X.values.reshape(X.shape[0], 28, 28)

# Sanity check of the converted data. 
print(sum(pd.DataFrame(X_train_tfd[1,:,:]).values.reshape(784) == X_train.iloc[1, :].to_numpy()) == 784)
```

Standardize the data.

```python
X_train_tfd = np.repeat((X_train_tfd / X_train_tfd.max()).astype("float32")[..., np.newaxis], 3, -1)
X_test_tfd = np.repeat((X_test_tfd / X_test_tfd.max()).astype("float32")[..., np.newaxis], 3, -1)
blind_test_tfd = np.repeat((blind_test_tfd / blind_test_tfd.max()).astype("float32")[..., np.newaxis], 3, -1)
X_tfd = np.repeat((X_tfd / X_tfd.max()).astype("float32")[..., np.newaxis], 3, -1)

print(X_train_tfd.shape)
print(X_test_tfd.shape)
print(blind_test_tfd.shape)
print(X_tfd.shape)
```

### Data Augmentation

```python
def random_reverse(x): 
    if np.random.random() > 0.5: 
        return x[:,::-1]
    else: 
        return x

def data_generator(X, Y, batch_size=100): 
    while True: 
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p, q = [], []
        for i in range(len(X)): 
            p.append(random_reverse(X[i]))
            q.append(Y[i])
            if len(p) == batch_size: 
                yield np.array(p), np.array(q)
                p, q = [], []
        if p: 
            yield np.array(p), np.array(q)
            p, q = [], []

from random_eraser import get_random_eraser
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=False))
```

### Define Training Model

```python
def build_model(): 
    input_image = Input(shape=(28, 28, 3))
    resized_image = Lambda(lambda image: resize_images(x=image, height_factor=2, width_factor=2, data_format='channels_last'))(input_image)

    base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=resized_image)
    # output = Dropout(0.5)(base_model.output)
    # predict = Dense(10, activation='softmax')(output)

    x = base_model.output
    # x = BatchNormalization()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### Model Training

#### Train the model with only training set. 

```python
history_bhp = model_bhp.fit_generator(datagen.flow(X_train_tfd, y_train, batch_size=32), 
                                      steps_per_epoch=X_train_tfd.shape[0], epochs=20, 
                                      validation_data=(X_test_tfd, y_test))
```

#### Train the model with both training and validation set for final preparation. 

```python
history_bhp_all = model_bhp.fit_generator(datagen.flow(X_tfd, y, batch_size=32), 
                                          steps_per_epoch=X_tfd.shape[0], epochs=8)
```

### Final Prediction

```python
blind_test_result = np.argmax(model_bhp.predict(blind_test_tfd), axis=-1)
blind_test_submission = pd.DataFrame([["Id", "predicted"]] + list(zip(range(0, len(blind_test_result)), blind_test_result)))
display(blind_test_submission[0:5])
blind_test_submission.to_csv("<output_file_name.csv>", index=False, header=False)
```

## Reference: 

1. Fashion-Mnist with MobileNet: [Blog](https://kexue.fm/archives/4556)
2. Random Erasing Data Augmentation: [Paper](https://arxiv.org/abs/1708.04896), [GitHub](https://github.com/zhunzhong07/Random-Erasing)