import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

train_vc_data = pd.read_csv('NEW_Vc.csv')
test_vh_data = pd.read_csv('NEW_Vh.csv')
train_groups = list(train_vc_data)
test_groups = list(test_vh_data)

vc_df = pd.DataFrame(train_vc_data)
train_indecies = vc_df.index
train_data = np.array(vc_df.values).astype("float")
#print(train_data)
vh_df = pd.DataFrame(test_vh_data)
test_indicies = vh_df.index
test_data = np.array(vh_df.values).astype("float")
#print(test_data)

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
print(train_data[0])  # Display sample features, notice the different scales

#Normalize the data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
mean = test_data.mean(axis=0)
std = test_data.std(axis=0)
test_data = (test_data - mean) / std

print("Normalized Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Normalized Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
print(train_data[0])  # Display sample features, notice the different scales

#Build the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

model = build_model()
model.summary()

#Print dots to display the training progress
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

#Store the training history
tf.placeholder
history = model.fit(train_data, train_indecies, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])

[loss, mae] = model.evaluate(test_data, test_indicies, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))