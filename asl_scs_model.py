import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

class CosSim2D(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        n_units=32,
        stride=1,
        depthwise_separable=False,
        padding='valid',
        q_init: float = 10,
        p_init: float = 1.,
        q_scale: float = .3,
        p_scale: float = 5,
        eps: float = 1e-6,
    ):
        super(CosSim2D, self).__init__()
        self.depthwise_separable = depthwise_separable
        self.n_units = n_units
        assert kernel_size in [1, 3, 5], "kernel of this size not supported"
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            self.stack = lambda x: x
        elif self.kernel_size == 3:
            self.stack = self.stack3x3
        elif self.kernel_size == 5:
            self.stack = self.stack5x5
        self.stride = stride
        if padding == 'same':
            self.pad = self.kernel_size // 2
            self.pad_1 = 1
            self.clip = 0
        elif padding == 'valid':
            self.pad = 0
            self.pad_1 = 0
            self.clip = self.kernel_size // 2
        self.p_init= p_init
        self.q_init= q_init
        self.p_scale = p_scale
        self.q_scale = q_scale
        self.eps = eps

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_y = math.ceil((self.in_shape[1] - 2*self.clip) / self.stride)
        self.out_x = math.ceil((self.in_shape[2] - 2*self.clip) / self.stride)
        self.flat_size = self.out_x * self.out_y
        self.channels = self.in_shape[3]

        if self.depthwise_separable:
            self.w = self.add_weight(
                shape=(1, tf.square(self.kernel_size), self.n_units),
                initializer="glorot_uniform", name='w',
                trainable=True,
            )
        else:
            self.w = self.add_weight(
                shape=(1, self.channels * tf.square(self.kernel_size), self.n_units),
                initializer="glorot_uniform", name='w',
                trainable=True,
            )

        # self.b = self.add_weight(
        #     shape=(self.n_units,), initializer="zeros", trainable=True, name='b')

        p_initializer = tf.keras.initializers.Constant(
            value=float(self.p_init * self.p_scale))
        q_initializer = tf.keras.initializers.Constant(
            value=float(self.q_init * self.q_scale))
        self.p = self.add_weight(
            shape=(self.n_units,),
            initializer=p_initializer,
            trainable=True,
            name='p')
        self.q = self.add_weight(
            shape=(1,),
            initializer=q_initializer,
            trainable=True,
            name='q')

    @tf.function
    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    @tf.function
    def sigplus(self, x):
        return tf.nn.sigmoid(x) * tf.nn.softplus(x)

    @tf.function
    def stack3x3(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(image[:, :y-1-self.clip:, :x-1-self.clip, :], tf.constant([[0,0], [self.pad,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],   # top row
                tf.pad(image[:, :y-1-self.clip, self.clip:x-self.clip, :],   tf.constant([[0,0], [self.pad,0], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-1-self.clip, 1+self.clip:, :],  tf.constant([[0,0], [self.pad,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                
                tf.pad(image[:, self.clip:y-self.clip, :x-1-self.clip, :],   tf.constant([[0,0], [0,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],   # middle row
                image[:,self.clip:y-self.clip:self.stride,self.clip:x-self.clip:self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 1+self.clip:, :],    tf.constant([[0,0], [0,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 1+self.clip:, :x-1-self.clip, :],  tf.constant([[0,0], [0,self.pad], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],    # bottom row
                tf.pad(image[:, 1+self.clip:, self.clip:x-self.clip, :],    tf.constant([[0,0], [0,self.pad], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:, 1+self.clip:, :],   tf.constant([[0,0], [0,self.pad], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:]
            ], axis=3)
        return stack
    
    @tf.function
    def stack5x5(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(image[:, :y-2-self.clip:, :x-2-self.clip, :],          tf.constant([[0,0], [self.pad,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 1:x-1-self.clip, :],         tf.constant([[0,0], [self.pad,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, self.clip:x-self.clip  , :], tf.constant([[0,0], [self.pad,0], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 1+self.clip:-1 , :],         tf.constant([[0,0], [self.pad,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 2+self.clip: , :],           tf.constant([[0,0], [self.pad,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
             
                tf.pad(image[:, 1:y-1-self.clip:,  :x-2-self.clip, :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:,  1:x-1-self.clip, :],         tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:,  self.clip:x-self.clip  , :], tf.constant([[0,0], [self.pad_1,self.pad_1], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:, 1+self.clip:-1  , :],         tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:, 2+self.clip:  , :],           tf.constant([[0,0], [self.pad_1,self.pad_1], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                
                tf.pad(image[:, self.clip:y-self.clip,  :x-2-self.clip, :],      tf.constant([[0,0], [0,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip,  1:x-1-self.clip, :],     tf.constant([[0,0], [0,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                       image[:, self.clip:y-self.clip,  self.clip:x-self.clip , :][:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 1+self.clip:-1  , :],     tf.constant([[0,0], [0,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 2+self.clip:  , :],       tf.constant([[0,0], [0,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 1+self.clip:-1,  :x-2-self.clip, :],           tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1,  1:x-1-self.clip, :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1,  self.clip:x-self.clip  , :],  tf.constant([[0,0], [self.pad_1,self.pad_1], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1, 1+self.clip:-1  , :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1, 2+self.clip:  , :],            tf.constant([[0,0], [self.pad_1,self.pad_1], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 2+self.clip:,  :x-2-self.clip, :],           tf.constant([[0,0], [0,self.pad], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:,  1:x-1-self.clip, :],          tf.constant([[0,0], [0,self.pad], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:,  self.clip:x-self.clip  , :],  tf.constant([[0,0], [0,self.pad], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:, 1+self.clip:-1  , :],          tf.constant([[0,0], [0,self.pad], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:, 2+self.clip:  , :],            tf.constant([[0,0], [0,self.pad], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
            ], axis=3)
        return stack

    def call_body(self, inputs):
        channels = tf.shape(inputs)[-1]
        x = self.stack(inputs)
        x = tf.reshape(
            x, (-1, self.flat_size, channels * tf.square(self.kernel_size)))
        p = tf.exp(self.p / self.p_scale)
        q = tf.exp(-self.q / self.q_scale)

        x_norm = (self.l2_normal(x, axis=2)) + q
        w_norm = (self.l2_normal(self.w, axis=1))
        x = tf.matmul(x / x_norm, self.w / w_norm)

        sign = tf.sign(x)
        x = tf.abs(x) + self.eps
        x = tf.pow(x, self.p)
        x = sign * x
        x = tf.reshape(x, (-1, self.out_y, self.out_x, self.n_units))
        return x

    @tf.function
    def call(self, inputs, training=None):
        if self.depthwise_separable:
            channels = tf.shape(inputs)[-1]
            x = tf.vectorized_map(self.call_body, tf.expand_dims(tf.transpose(inputs, (3,0,1,2)), axis=-1))
            s = tf.shape(x)
            x = tf.transpose(x, (1,2,3,4,0))
            x = tf.reshape(x, (-1, self.out_y, self.out_x, self.channels * self.n_units))
            return x
        else:
            x = self.call_body(inputs)
            return x


##############################################################################
import tensorflow as tf

class MaxAbsPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size, pad_to_fit=False):
        super(MaxAbsPool2D, self).__init__()
        self.pad = pad_to_fit
        self.pool_size = pool_size

    def compute_output_shape(self, in_shape):
        if self.pad:
            return (in_shape[0],
                    tf.math.ceil(in_shape[1] / self.pool_size),
                    tf.math.ceil(in_shape[2] / self.pool_size),
                    in_shape[3])
        return (in_shape[0],
                (in_shape[1] // self.pool_size),
                (in_shape[2] // self.pool_size),
                in_shape[3])

    def compute_padding(self, in_shape):
        mod_y = in_shape[1] % self.pool_size
        y1 = mod_y // 2
        y2 = mod_y - y1
        mod_x = in_shape[2] % self.pool_size
        x1 = mod_x // 2
        x2 = mod_x - x1
        self.padding = ((0, 0), (y1, y2), (x1, x2), (0, 0))

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_shape = self.compute_output_shape(self.in_shape)
        self.compute_padding(self.in_shape)

    def stack(self, inputs):
        if self.pad:
            inputs = tf.pad(inputs, self.padding)
        max_height = (tf.shape(inputs)[1] // self.pool_size) * self.pool_size
        max_width = (tf.shape(inputs)[2] // self.pool_size) * self.pool_size
        stack = tf.stack(
            [inputs[:, i:max_height:self.pool_size, j:max_width:self.pool_size, :]
             for i in range(self.pool_size) for j in range(self.pool_size)],
            axis=-1)
        return stack

    @tf.function
    def call(self, inputs, training=None):
        stacked = self.stack(inputs)
        inds = tf.argmax(tf.abs(stacked), axis=-1, output_type=tf.int32)
        ks = tf.shape(stacked)
        idx = tf.stack([
            *tf.meshgrid(
                tf.range(0, ks[0]),
                tf.range(0, ks[1]),
                tf.range(0, ks[2]),
                tf.range(0, ks[3]),
                indexing='ij'
            ), inds],
            axis=-1)
        x = tf.gather_nd(stacked, idx)
        x = tf.reshape(x, (-1, *self.out_shape[1:]))
        return x


##############################################################################
import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Activation, Conv2D, Dense, Dropout, Flatten,
    InputLayer, MaxPool2D, Resizing, Rescaling)
from sklearn import metrics
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# from sharpened_cosine_similarity import CosSim2D
# from max_abs_pool import MaxAbsPool2D

n_units = 16
CLASSES = 29
LEARNING_RATE = 0.01
LABEL_SMOOTHING = 0.05
EPOCHS = 40
p_init = .7

LABELS_COLUMN = "Label"
FILENAMES_COLUMN = "Filename"

BATCH_SIZE = 128
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CHANNELS = 3

TRAINING_DIR = "asl_alphabet_train/asl_alphabet_train"
TESTING_DIR = "asl_alphabet_test/asl_alphabet_test"

labels = [label for label in os.listdir(TRAINING_DIR)]
df = pd.DataFrame([
    {
        LABELS_COLUMN: label,
        FILENAMES_COLUMN: os.path.join(TRAINING_DIR, label, filename)
    }
    for label in labels
    for filename in os.listdir(os.path.join(TRAINING_DIR, label))
])
df = df.sample(frac=1).reset_index(drop=True)

f_v = .2  # Validation fraction
f_p = .1  # Holdout fraction
dv = df.sample(frac=f_v)
df=df.drop(dv.index)
dp = df.sample(frac=f_p)
dt=df.drop(dp.index)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("Devices: ", tf.config.list_physical_devices("GPU"))

print(f"Length of training images list: {len(dt)}.")
print(f"Length of validation images list: {len(dv)}.")
print(f"Length of prediction images list: {len(dp)}.")
print(f"Number of unique labels in training images list: {len(dt[LABELS_COLUMN].unique())}.")
print(f"Number of unique labels in validation images list: {len(dv[LABELS_COLUMN].unique())}.")
print(f"Number of unique labels in prediction images list: {len(dp[LABELS_COLUMN].unique())}.")

xyt = ImageDataGenerator().flow_from_dataframe(
    dt,
    x_col=FILENAMES_COLUMN,
    y_col=LABELS_COLUMN,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode='categorical',
    batch_size=BATCH_SIZE
)
xyv = ImageDataGenerator().flow_from_dataframe(
    dv,
    x_col=FILENAMES_COLUMN,
    y_col=LABELS_COLUMN,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)
xyp = ImageDataGenerator().flow_from_dataframe(
    dp,
    x_col=FILENAMES_COLUMN,
    y_col=LABELS_COLUMN,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

x, y = next(xyt)
class_labels = list(xyt.class_indices.keys())
class_labels = ["Label: " + class_labels[c] for c in np.argmax(y[0:16], axis=-1)]


model = Sequential([
    InputLayer((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), name="input"),

    CosSim2D(kernel_size=5, n_units=n_units, p_init=p_init),
    MaxAbsPool2D(2),

    CosSim2D(kernel_size=3, n_units=n_units, p_init=p_init),
    MaxAbsPool2D(2),

    CosSim2D(kernel_size=3, n_units=n_units, p_init=p_init),
    MaxAbsPool2D(2),

    CosSim2D(kernel_size=3, n_units=n_units, p_init=p_init),
    MaxAbsPool2D(4),

    Flatten(name="flatten"),
    Dense(CLASSES, name="dense"),
], name="ASL")

model.summary()

optimizer = Adam(learning_rate=LEARNING_RATE)
loss = tf.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=LABEL_SMOOTHING)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

VERBOSE = 1
callbacks = [
    LambdaCallback(
        on_epoch_end=lambda batch, logs: time.sleep(15),
    )
]
history = model.fit(
    xyt,
    validation_data=xyv,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=VERBOSE)

p = model.predict(xyp, verbose=1)
report = metrics.classification_report(
    xyp.classes,
    np.argmax(p, axis=-1),
    target_names=list(xyp.class_indices.keys())
)
print(report)

import plotly.express as px

def plot_confusion_matrix(confusion_matrix, labels):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z = confusion_matrix,
            x = labels,
            y = labels,
        )
    )
    return fig

cm = metrics.confusion_matrix(xyp.classes, np.argmax(p, axis=-1))
fig = plot_confusion_matrix(cm, list(xyp.class_indices.keys()))
fig.update_layout(width=800, height=800)
fig.show()

from keras import backend as K
print(model.optimizer.learning_rate.numpy())
K.set_value(model.optimizer.learning_rate, 0.0001)
print(model.optimizer.learning_rate.numpy())
