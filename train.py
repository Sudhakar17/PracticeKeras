from keras.callbacks import ModelCheckpoint,TensorBoard, LearningRateScheduler, CSVLogger
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import (ImageDataGenerator,apply_affine_transform,
                                       random_brightness, apply_channel_shift,
                                       random_channel_shift)
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import CNN

# import resnet

import numpy as np
import pandas as pd


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 20))

train = pd.read_csv("D:\\1_Datasets\\digit-recognizer\\train.csv")
test = pd.read_csv("D:\\1_Datasets\\digit-recognizer\\test.csv")

# set the random seed and hyperparams
random_seed = 42
epochs = 60
batch_size = 64
lr=0.001
checkpointer = ModelCheckpoint(filepath="./weights/cnn_weights_l2.hdf5", 
                    verbose=1, save_best_only=True, save_weights_only=True)

# training and validation dataset preparation
y_train = train["label"]
x_train = train.drop(labels = ["label"],axis = 1) # Drop 'label' column

# Normalize the data
x_train = x_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 1 -> [0,1,0,0,0,0,0,0,0,0])
# y_train = to_categorical(y_train, num_classes = 10)

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, random_state=42)



train_datag_1 = ImageDataGenerator(featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10.,
                                   preprocessing_function=apply_affine_transform
                                   )


# train_datag_1_flow = train_datag_1.flow(X_train, Y_train, batch_size=batch_size)
# train_datag_2_flow = train_datag_2.flow(X_train, Y_train, batch_size=batch_size)
# train_generator = zip(train_datag_1_flow, train_datag_2_flow)

val_datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10.,
                                 preprocessing_function=apply_affine_transform
                                 )


train_datag_1.fit(X_train)
val_datagen.fit(X_val)
model = CNN.model()

# model = resnet.ResnetBuilder.build_resnet_18((32,32,3),NUM_CLASSES)

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


lr_scheduler = LearningRateScheduler(lr_schedule)


model.fit_generator(train_datag_1.flow(X_train, Y_train,
                    batch_size=batch_size),
                    steps_per_epoch=(X_train.shape[0]*2) // batch_size,
                    epochs=epochs,
                    validation_data=val_datagen.flow(X_val, Y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks=[checkpointer, lr_scheduler])

score = model.evaluate(X_val, Y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

# predict results in the test set
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_v3.csv",index=False)

