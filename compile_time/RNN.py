# #reference:
# #https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/guide/keras/rnn.ipynb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import timeit
input_dim = 28

units = 64
output_size = 10
# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model

model = build_model(allow_cudnn_kernel=True)
@tf.function(jit_compile=True)
def test(input):
    # classifier_model.compile(optimizer=optimizer,
    #                      loss=loss,
    #                      metrics=metrics,jit_compile=True)
    # classifier_model.add(Dense(10, activation = 'softmax'))
    model(input)
m=0
TIMES=3
i1 = 32
i2 = 64
i3 = 32
batch_size = 64
num_batches = 10
timestep = 50
m=[]
for i in range(TIMES):
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]
    n=timeit.timeit(lambda:test(tf.expand_dims(sample, 0)), number=10)
    m.append(n)
    
    print("execution time of RNN model",n)
print("compilation time of RNN model",m[0]-m[1])
    
    
