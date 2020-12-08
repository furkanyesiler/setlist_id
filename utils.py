import keras as K
import librosa
import numpy as np
import os
import scipy


def pairwise_euclidean_distance(x, y=None, eps=1e-12):
    """
    Computing squared Euclidean distances between the elements of two tensors.

    :param x: first tensor
    :param y: second tensor (optional)
    :param eps: epsilon value for avoiding div by zero
    :return: pairwise distance matrix
    """
    x_norm = np.power(x, 2).sum(1).reshape(-1, 1)
    if y is not None:
        y_norm = np.power(y, 2).sum(1).reshape(1, -1)
    else:
        y = x
        y_norm = x_norm.reshape(1, -1)

    dist = x_norm + y_norm - 2 * np.matmul(x, y.T)
    return dist + eps


def pairwise_cosine_distance(x, y=None):
    """
    Computing cosine similarity between the elements of two tensors.

    :param x: first tensor
    :param y: second tensor (optional)
    :return: pairwise similarity matrix
    """
    if y is None:
        y = x
    return -1 * np.divide(np.divide(x @ y.T, np.linalg.norm(y, axis=1)),
                          np.linalg.norm(x, axis=1).reshape(-1, 1))


def chrompwr(chroma_feat, p=.5):
    """
    Raise chroma columns to a power, preserving norm

    :param chroma_feat: chroma feature
    :param p: power
    :return: transformed chroma feature

    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    Taken from:
    https://github.com/urinieto/LargeScaleCoverSongId/blob/master/dan_tools.py
    """
    nchr, nbts = chroma_feat.shape
    # norms of each input col
    c_mn = np.tile(np.sqrt(np.sum(chroma_feat * chroma_feat, axis=0)),
                   (nchr, 1))
    c_mn[c_mn == 0] = 1
    # normalize each input col, raise to power
    c_mp = np.power(chroma_feat/c_mn, p)
    # norms of each resultant column
    c_mpn = np.tile(np.sqrt(np.sum(c_mp * c_mp, axis=0)), (nchr, 1))
    c_mpn[np.where(c_mpn == 0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return c_mn * (c_mp / c_mpn)


def btchroma_to_fftmat(btchroma, win=75):
    """
    Stack the flattened result of fft2 on patches 12 x win

    :param btchroma: chroma feature
    :param win: window size for patches
    :return: 2d-fft magnitudes for the patches

    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    Taken from:
    https://github.com/urinieto/LargeScaleCoverSongId/blob/master/dan_tools.py
    """
    # 12 semitones
    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        F = scipy.fftpack.fft2(btchroma[:, i:i+win])
        F = np.sqrt(np.real(F)**2 + np.imag(F)**2)
        patch = scipy.fftpack.fftshift(F)
        fftmat[:, i] = patch.flatten()
    return fftmat


def extract_2dftm(feat_s, beats_s):
    """
    Extract 2DFTM embedding for a given chroma feature and beats

    :param feat_s: input chroma feature
    :param beats_s: input beats
    :return: 2DFTM embedding of the input
    """
    if beats_s.size < 75:
        if feat_s.shape[1] < 75:
            crema_s = np.concatenate((feat_s,
                                      np.zeros((feat_s.shape[0],
                                                75 - feat_s.shape[1]))),
                                     axis=1)
        else:
            crema_s = feat_s
    else:
        crema_s = librosa.util.sync(feat_s, beats_s, aggregate=np.median)
    chroma_s = chrompwr(crema_s, 1.96)
    # Get all 2D FFT magnitude shingles
    shingles = btchroma_to_fftmat(chroma_s, 75).T
    Norm = np.sqrt(np.sum(shingles ** 2, 1))
    Norm[Norm == 0] = 1
    shingles = np.log(5 * shingles / Norm[:, None] + 1)
    shingle = np.median(shingles, 0)  # Median aggregate
    shingle = shingle / np.sqrt(np.sum(shingle ** 2))
    shingle = shingle[np.newaxis, :]
    return shingle


def create_re_move_model():
    """
    Create Keras version of the Re-MOVE model

    :return: Re-MOVE model
    """
    inp = K.Input(shape=(None, 23, 1), dtype=np.float32)
    c1 = K.layers.Convolution2D(256, (180, 12), padding='valid',
                                activation=None,
                                data_format='channels_last')(inp)
    p1 = CustomPReLU()(c1)
    mp = K.layers.MaxPooling2D(pool_size=(1, 12))(p1)
    c2 = K.layers.Convolution2D(256, (5, 1), padding='valid',
                                activation=None,
                                data_format='channels_last')(mp)
    p2 = CustomPReLU()(c2)
    c3 = K.layers.Convolution2D(256, (5, 1), dilation_rate=(20, 1),
                                padding='valid', activation=None,
                                data_format='channels_last')(p2)
    p3 = CustomPReLU()(c3)
    c4 = K.layers.Convolution2D(256, (5, 1), padding='valid', activation=None,
                                data_format='channels_last')(p3)
    p4 = CustomPReLU()(c4)
    c5 = K.layers.Convolution2D(512, (5, 1), dilation_rate=(13, 1),
                                padding='valid', activation=None,
                                data_format='channels_last')(p4)
    p5 = CustomPReLU()(c5)
    feat = AutoPool()(p5)
    sque = K.layers.Reshape((256,))(feat)
    lin = K.layers.Dense(256, use_bias=False)(sque)
    bn = K.layers.BatchNormalization(center=False, scale=False)(lin)
    re_move = K.models.Model(inp, [bn])
    re_move.load_weights(os.path.join(os.path.dirname(__file__),
                                      're-move_256_keras.h5'))

    return re_move


class CustomPReLU(K.layers.Layer):
    """
    Custom Parametric ReLU layer
    """
    def __init__(self, **kwargs):
        super(CustomPReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = tuple(1 for i in range(len(input_shape) - 1)) + \
                      input_shape[-1:]
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=K.initializers.get('zeros'),
                                     regularizer=K.regularizers.get(None),
                                     constraint=K.constraints.get(None))
        self.built = True

    def call(self, inputs):
        pos = K.activations.relu(inputs)
        neg = -self.alpha * K.activations.relu(-inputs)
        return pos + neg

    def compute_output_shape(self, input_shape):
        return input_shape


class AutoPool(K.layers.Layer):
    """
    Custom AutoPool layer
    """
    def __init__(self, **kwargs):
        super(AutoPool, self).__init__(**kwargs)

    def build(self, input_shape):
        init = K.initializers.get('zeros')
        reg = K.regularizers.get(None)
        const = K.constraints.get(None)
        self.autopool = self.add_weight(shape=(1),
                                        name='autopool',
                                        initializer=init,
                                        regularizer=reg,
                                        constraint=const)
        self.built = True

    def call(self, inputs):
        output = K.layers.Lambda(lambda x: self.autopool_w(x))(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 256)

    def autopool_w(self, input):
        i1 = input[:, :, :, :256]
        i2 = input[:, :, :, 256:]

        x = i1 * self.autopool
        max_values = K.backend.max(x, axis=1, keepdims=True)
        softmax = K.backend.exp(x - max_values)
        weights = softmax / K.backend.sum(softmax, axis=1, keepdims=True)
        return K.backend.squeeze(K.backend.sum((weights * i2), axis=1), 1)

