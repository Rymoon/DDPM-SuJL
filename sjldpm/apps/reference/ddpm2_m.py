
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback
from keras.initializers import VarianceScaling
from keras_preprocessing.image import list_pictures
from bert4keras.layers import ScaleOffset
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_layer_adaptation
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_exponential_moving_average
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告


class GroupNorm(ScaleOffset):
    """定义GroupNorm，默认groups=32
    """
    def call(self, inputs):
        inputs = K.reshape(inputs, (-1, 32), -1)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        inputs = (inputs - mean) * tf.rsqrt(variance + 1e-6)
        inputs = K.flatten(inputs, -2)
        return super(GroupNorm, self).call(inputs)


def dense(x, out_dim, activation=None, init_scale=1):
    """Dense包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Dense(
        out_dim,
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)


def conv2d(x, out_dim, activation=None, init_scale=1):
    """Conv2D包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Conv2D(
        out_dim, (3, 3),
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)



def residual_block(x, ch, t, embedding_size ):
    """残差block
    """
    in_dim = K.int_shape(x)[-1]
    out_dim = ch * embedding_size
    if in_dim == out_dim:
        xi = x
    else:
        xi = dense(x, out_dim)
    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = conv2d(x, out_dim)
    x = Add()([x, dense(t, K.int_shape(x)[-1])])
    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = conv2d(x, out_dim, None, 0)
    x = Add()([x, xi])
    return x

def l2_loss__vec(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return K.sum((y_true - y_pred)**2, axis=[1, 2])

def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return K.sum((y_true - y_pred)**2, axis=[1, 2, 3])

def residual_block__vec(x, ch, t, embedding_size):
    """
    Use dense instead of conv2d
    N: batch_size, as None in tf_shape
    L: vector-length
    I: in_dim, channel of input
    O: out_dim, channel of output
    """
    in_dim = K.int_shape(x)[-1]
    out_dim = ch * embedding_size
    if in_dim == out_dim:
        xi = x
    else:
        xi = dense(x, out_dim)
    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = dense(x, out_dim) # NLO
    x = Add()([x, dense(t, K.int_shape(x)[-1])])
    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = dense(x, out_dim, None, 0)
    x = Add()([x, xi])

    return x

def get_model(*,resize_size, embedding_size, channels, blocks, T):
    """
    Return compiled model;
    
    model.optimizer is valied;
    """
    # 搭建去噪模型
    x_in = x = Input(shape=(*resize_size, 3))
    t_in = Input(shape=(1,))
    t = Embedding(
        input_dim=T,
        output_dim=embedding_size,
        embeddings_initializer='Sinusoidal',
        trainable=False
    )(t_in)
    t = dense(t, embedding_size * 4, 'swish')
    t = dense(t, embedding_size * 4, 'swish')
    t = Lambda(lambda t: t[:, None])(t)

    x = conv2d(x, embedding_size)
    inputs = [x]

    # UNet-residual
    for i, ch in enumerate(channels):
        for j in range(blocks):
            x = residual_block(x, ch, t, embedding_size)
            inputs.append(x)
        if i != len(channels) - 1:
            x = AveragePooling2D((2, 2))(x)
            inputs.append(x)

    x = residual_block(x, ch, t, embedding_size)

    for i, ch in enumerate(channels[::-1]):
        for j in range(blocks + 1):
            x = Concatenate()([x, inputs.pop()])
            x = residual_block(x, ch, t, embedding_size)
        if i != len(channels) - 1:
            x = UpSampling2D((2, 2))(x)

    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = conv2d(x, 3)

    model = Model(inputs=[x_in, t_in], outputs=x)
    model.summary()

    OPT = extend_with_layer_adaptation(Adam)
    OPT = extend_with_piecewise_linear_lr(OPT)  # 此时就是LAMB优化器
    OPT = extend_with_exponential_moving_average(OPT)  # 加上滑动平均
    optimizer = OPT(
        learning_rate=1e-3,
        ema_momentum=0.9999,
        exclude_from_layer_adaptation=['Norm', 'bias'],
        lr_schedule={
            4000: 1,  # Warmup步数
            20000: 0.5,
            40000: 0.1,
        }
    )
    model.compile(loss=l2_loss, optimizer=optimizer)

    return model

def get_model__vec(*, walker_dim =1, walker_ch = 1,embedding_size, channels, blocks, T):
    """
    Use FNN for vector input-output

    N=batch_size, as None in tf_shape.
    L=walker_dim
    E=embedding_size
    C=walker_ch
    """
    x_in = x = Input(shape=(walker_dim,  walker_ch ))
    t_in = Input(shape=(1,))
    t = Embedding(
        input_dim=T,
        output_dim=embedding_size,
        embeddings_initializer='Sinusoidal',
        trainable=False
    )(t_in) # (None, 1, embedding_size)
    t = dense(t, embedding_size * 4, 'swish')
    t = dense(t, embedding_size * 4, 'swish')
    # t = Lambda(lambda t: t[:, None])(t) # for vec 3d-tensor is ok; for img 4d-tensorn eed this line

    x = dense(x, embedding_size) # N L E
    inputs = [x]

    for i,ch in enumerate(channels):
        for j in range(blocks):
            x = residual_block__vec(x, ch, t, embedding_size)
            inputs.append(x)
        if i != len(channels) - 1:
            inputs.append(x)
    
    x = residual_block__vec(x, ch, t, embedding_size)

    for i, ch in enumerate(channels[::-1]):
        for j in range(blocks + 1):
            x = Concatenate()([x, inputs.pop()])
            x = residual_block__vec(x, ch, t, embedding_size)
    assert len(inputs)==0
    x = GroupNorm()(x)
    x = Activation('swish')(x)
    x = dense(x,  walker_ch )

    model = Model(inputs=[x_in, t_in], outputs=x)
    model.summary()

    OPT = extend_with_layer_adaptation(Adam)
    OPT = extend_with_piecewise_linear_lr(OPT)  # 此时就是LAMB优化器
    OPT = extend_with_exponential_moving_average(OPT)  # 加上滑动平均
    optimizer = OPT(
        learning_rate=1e-3,
        ema_momentum=0.9999,
        exclude_from_layer_adaptation=['Norm', 'bias'],
        lr_schedule={
            4000: 1,  # Warmup步数
            20000: 0.5,
            40000: 0.1,
        }
    )
    model.compile(loss=l2_loss__vec, optimizer=optimizer)

    return model


class Trainer(Callback):
    """训练回调器
    
    self.sample = sample(model, path)
    """
    def __init__(self, model, optimizer, sample, logdir):
        self.model = model
        self.optimizer = optimizer
        self.sample = sample
        self.logdir= logdir
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(Path(self.logdir,f"weights/model.weights").as_posix())
        self.sample(self.model, Path(self.logdir,f"samples/{(epoch+1):05d}.png").as_posix())
        self.optimizer.apply_ema_weights()
        self.model.save_weights('model.ema.weights')
        self.sample(self.model, Path(self.logdir,f"samples/{(epoch+1):05d}.png").as_posix())
        self.optimizer.reset_old_weights()
        
        print(" - log_dir: ",Path(self.logdir).as_posix())


class Trainer__vec(Callback):
    """训练回调器
    
    self.sample = sample(model, path)
    """
    def __init__(self, model, optimizer, sample, logdir):
        self.model = model
        self.optimizer = optimizer
        self.sample = sample
        self.logdir= logdir
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(Path(self.logdir,f"weights/model.weights").as_posix())
        self.sample(self.model, Path(self.logdir,f"samples/{(epoch+1):05d}.npy").as_posix())
        self.optimizer.apply_ema_weights()
        self.model.save_weights('model.ema.weights')
        self.sample(self.model, Path(self.logdir,f"samples/{(epoch+1):05d}.npy").as_posix())
        self.optimizer.reset_old_weights()