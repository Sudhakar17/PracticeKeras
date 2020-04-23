import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

# Load the data
train = pd.read_csv("D:\\Dataset\\digit-recognizer\\train.csv")
test = pd.read_csv("D:\\Dataset\\digit-recognizer\\test.csv")


# set the random seed and hyperparams
random_seed = 42
epochs = 50
batch_size = 32
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
checkpointer = ModelCheckpoint(filepath="./weights/cnn_weights.hdf5", 
                    verbose=1, save_best_only=True, save_weights_only=True)

# plots
def histogram_of_digits():
    plt.figure(figsize=(12,8))
    ax = sns.countplot(y_train)
    plt.title('Frequency count of digits')
    plt.xlabel('Digits')
    plt.show()

# confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# training and validation dataset preparation
y_train = train["label"]
x_train = train.drop(labels = ["label"],axis = 1) # Drop 'label' column
# histogram_of_digits()
# print(y_train.value_counts())


# Normalize the data
x_train = x_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 1 -> [0,1,0,0,0,0,0,0,0,0])
y_train = to_categorical(y_train, num_classes = 10)

# Split the train and the validation set for the fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, 
                            random_state=random_seed)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False)

datagen.fit(x_train)


# CNN model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()



# compile a model
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=adam,metrics=['accuracy'])

# Fit the model
history = model.fit_generator(datagen.flow(x_train, y_train,
          batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          callbacks = [checkpointer],
          validation_data=(x_val, y_val))



# # Predict the values from the validation dataset
# y_pred = model.predict(x_val)
# # Convert predictions classes to one hot vectors 
# y_pred_classes = np.argmax(y_pred,axis = 1) 
# # Convert validation observations to one hot vectors
# y_true = np.argmax(y_val,axis = 1) 
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(y_true, y_pred_classes)
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = range(10))


score = model.evaluate(x_val, y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

# predict results in the test set
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_v1.csv",index=False)
