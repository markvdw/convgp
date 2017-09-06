# Copyright 2017 Mark van der Wilk, James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

import GPflow

float_type = GPflow.settings.dtypes.float_type


class Conv(GPflow.kernels.Kern):
    """
    Conv
    Plain convolutional kernel.
    """

    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        GPflow.kernels.Kern.__init__(self, np.prod(img_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        self.basekern.input_dim = np.prod(patch_size)
        self.colour_channels = colour_channels

    def _get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_size)
        """
        # castX = tf.transpose(
        #     tf.reshape(tf.cast(X, tf.float32, name="castX"), tf.stack([tf.shape(X)[0], -1, self.colour_channels])),
        #     [0, 2, 1])
        # castX = tf.cast(X, tf.float32, name="castX")

        # Roll the colour channel to the front, so it appears to `tf.extract_image_patches()` as separate images. Then
        # extract patches and reshape to have the first axis the same as the number of images. The separate patches will
        # then be in the second axis.
        castX = tf.transpose(
            tf.reshape(X, tf.stack([tf.shape(X)[0], -1, self.colour_channels])),
            [0, 2, 1])
        patches = tf.extract_image_patches(
            tf.reshape(castX, [-1, self.img_size[0], self.img_size[1], 1], name="rX"),
            [1, self.patch_size[0], self.patch_size[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        return tf.cast(tf.reshape(patches, [tf.shape(X)[0], self.colour_channels * shp[1] * shp[2], shp[3]]),
                       float_type)

    def K(self, X, X2=None):
        Xp = self._get_patches(X)
        Xp = tf.reshape(Xp, (-1, self.patch_len))
        Xp2 = tf.reshape(self._get_patches(X2), (tf.shape(X2)[0], self.patch_len)) if X2 is not None else None

        bigK = self.basekern.K(Xp, Xp2)  # N * num_patches x N * num_patches
        K = tf.reduce_sum(tf.reshape(bigK, (tf.shape(X)[0], self.num_patches, -1, self.num_patches)), [1, 3])
        return K / self.num_patches ** 2.0

    def Kdiag(self, X):
        Xp = self._get_patches(X)

        def sumbK(Xp):
            return tf.reduce_sum(self.basekern.K(Xp))

        return tf.map_fn(sumbK, Xp) / self.num_patches ** 2.0
        # return tf.reduce_sum(tf.map_fn(self.basekern.K, Xp), [1, 2]) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        Xp = self._get_patches(X)  # N x num_patches x patch_len
        Xp = tf.reshape(Xp, (-1, self.patch_len))
        bigKzx = self.basekern.K(Z, Xp)  # M x N * num_patches
        Kzx = tf.reduce_sum(tf.reshape(bigKzx, (tf.shape(Z)[0], tf.shape(X)[0], self.num_patches)), [2])
        return Kzx / self.num_patches

    def Kzz(self, Z):
        return self.basekern.K(Z)

    def init_inducing(self, X, M, method="default"):
        if method == "default" or method == "random":
            patches = self.compute_patches(X[np.random.permutation(len(X))[:M], :]).reshape(-1, self.patch_len)
            Zinit = patches[np.random.permutation(len(patches))[:M], :]
            Zinit += np.random.rand(*Zinit.shape) * 0.001
            return Zinit
        elif method == "patches-unique":
            patches = np.unique(self.compute_patches(
                X[np.random.permutation(len(X))[:M], :]).reshape(-1, self.patch_len), axis=0)
            return patches[np.random.permutation((len(patches)))[:M], :]
        else:
            raise NotImplementedError

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return (self.img_size[0] - self.patch_size[0] + 1) * (
            self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels

    @GPflow.param.AutoFlow((float_type,))
    def compute_patches(self, X):
        return self._get_patches(X)

    @GPflow.param.AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_Kzx(self, Z, X):
        return self.Kzx(Z, X)


class ColourPatchConv(Conv):
    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        Conv.__init__(self, basekern, img_size, patch_size, colour_channels)
        self.basekern.input_dim = np.prod(patch_size) * self.colour_channels

    def _get_patches(self, X):
        """
        Extracts patches from the images X.
        :param X: (N x img_size_flattened, channels)
        :return: Patches (N, num_patches, patch_size)
        """
        castX = tf.cast(X, tf.float32, name="castX")
        patches = tf.extract_image_patches(
            tf.reshape(castX, [tf.shape(X)[0], self.img_size[0], self.img_size[1], self.colour_channels], name="rX"),
            [1, self.patch_size[0], self.patch_size[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        return tf.cast(tf.reshape(patches, [tf.shape(X)[0], shp[1] * shp[2], shp[3]]), float_type)

    @property
    def patch_len(self):
        return np.prod(self.patch_size) * self.colour_channels

    @property
    def num_patches(self):
        return (self.img_size[0] - self.patch_size[0] + 1) * (self.img_size[1] - self.patch_size[1] + 1)


class WeightedConv(Conv):
    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        Conv.__init__(self, basekern, img_size, patch_size, colour_channels)
        self.W = GPflow.param.Param(np.ones(self.num_patches))

    def K(self, X, X2=None):
        Xp = self._get_patches(X)
        Xp = tf.reshape(Xp, (-1, self.patch_len))
        Xp2 = None if X2 is None else tf.reshape(self._get_patches(X2),
                                                 (tf.shape(X2)[0] * self.num_patches, self.patch_len))

        bigK = self.basekern.K(Xp, Xp2)  # N * num_patches x N * num_patches
        bigK = tf.reshape(bigK, (tf.shape(X)[0], self.num_patches, -1, self.num_patches))  # N x numpatch x M x numpatch

        W2 = tf.expand_dims(self.W, 0) * tf.expand_dims(self.W, 1)
        W2 = tf.expand_dims(W2, 0)
        W2 = tf.expand_dims(W2, 2)
        W2bigK = bigK * W2
        K = tf.reduce_sum(W2bigK, [1, 3]) / self.num_patches ** 2.0
        return K

    def Kdiag(self, X):
        Xp = self._get_patches(X)  # N x num_patches x patch_dim

        W2 = tf.expand_dims(self.W, 0) * tf.expand_dims(self.W, 1)

        def Kdiag_element(patches):
            return tf.reduce_sum(self.basekern.K(patches) * W2)

        return tf.map_fn(Kdiag_element, Xp) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        Xp = self._get_patches(X)  # N x num_patches x patch_len

        Xp = tf.reshape(Xp, (-1, self.patch_len))
        bigKzx = self.basekern.K(Z, Xp)  # M x N * num_patches
        bigKzx = tf.reshape(bigKzx, (tf.shape(Z)[0], tf.shape(X)[0], self.num_patches))
        Kzx = tf.reduce_sum(bigKzx * self.W, [2])
        return Kzx / self.num_patches

        # def Kzx_n(patches):
        #     return tf.reduce_sum(self.basekern.K(Z, patches) * self.W, 1) / self.num_patches
        #
        # return tf.transpose(tf.map_fn(Kzx_n, Xp))


class WeightedColourPatchConv(ColourPatchConv, WeightedConv):
    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        WeightedConv.__init__(self, basekern, img_size, patch_size, colour_channels)
        ColourPatchConv.__init__(self, basekern, img_size, patch_size, colour_channels)


class ConvRBF(Conv):
    def __init__(self, img_size, patch_size):
        base = GPflow.kernels.RBF(np.prod(patch_size), ARD=False)
        Conv.__init__(self, base, img_size, patch_size)
        # tf.conv2d does not have a float64 version implemented yet, so keep this bit of code locally float32.
        self.conv_dtype = tf.float32

    def Kzx(self, Z, X):
        X = tf.reshape(X, [-1, self.img_size[0], self.img_size[1], 1], name="X")
        Z = tf.reshape(Z, [-1, self.patch_size[0], self.patch_size[1]], name="Z")
        blank_patch = tf.ones([self.patch_size[0], self.patch_size[1], 1, 1], dtype=self.conv_dtype)
        striding = [1, 1, 1, 1]
        xTx = tf.nn.conv2d(tf.cast(tf.square(X), self.conv_dtype), blank_patch, striding, 'VALID',
                           name="xTx")  # N x ~D x ~D
        convZ = tf.cast(tf.expand_dims(tf.transpose(Z, [1, 2, 0]), 2), self.conv_dtype, name="convZ")
        xTz = tf.nn.conv2d(tf.cast(X, self.conv_dtype), convZ, striding, 'VALID', name="xTz")  # N x ~D x ~D x M
        zTz = tf.reduce_sum(tf.square(Z), [1, 2])  # M,

        arg = -(tf.cast(xTx, float_type) - 2. * tf.cast(xTz, float_type) + tf.reshape(zTz, [1, 1, 1, -1])) / tf.square(
            self.basekern.lengthscales)
        bigK = self.basekern.variance * tf.exp(arg / 2)
        return tf.transpose(tf.reduce_sum(bigK, [1, 2])) / self.num_patches


class WeightedConvRBF(ConvRBF):
    def __init__(self, img_size, patch_size):
        ConvRBF.__init__(self, img_size, patch_size)
        self.W = GPflow.param.Param(np.ones(self.num_patches))

    def Kzx(self, Z, X):
        X = tf.reshape(X, [-1, self.img_size[0], self.img_size[1], 1], name="X")
        Z = tf.reshape(Z, [-1, self.patch_size[0], self.patch_size[1]], name="Z")
        blank_patch = tf.ones([self.patch_size[0], self.patch_size[1], 1, 1], dtype=self.conv_dtype)
        striding = [1, 1, 1, 1]
        xTx = tf.nn.conv2d(tf.cast(tf.square(X), self.conv_dtype), blank_patch, striding, 'VALID',
                           name="xTx")  # N x ~D x ~D
        convZ = tf.cast(tf.expand_dims(tf.transpose(Z, [1, 2, 0]), 2), self.conv_dtype, name="convZ")
        xTz = tf.nn.conv2d(tf.cast(X, self.conv_dtype), convZ, striding, 'VALID', name="xTz")  # N x ~D x ~D x M
        zTz = tf.reduce_sum(tf.square(Z), [1, 2])  # M,

        arg = -(tf.cast(xTx, float_type) - 2. * tf.cast(xTz, float_type) + tf.reshape(zTz, [1, 1, 1, -1])) / tf.square(
            self.basekern.lengthscales)
        bigK = self.basekern.variance * tf.exp(arg / 2)
        W = tf.reshape(self.W, tf.concat([[1], tf.shape(bigK)[1:3], [1]], axis=0))
        return tf.transpose(tf.reduce_sum(bigK * W, [1, 2])) / self.num_patches
