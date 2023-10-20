import tensorflow as tf

from typing import cast, Dict, Optional, Sequence, Tuple, Union
from typing import Union, Text, Optional

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                arch,
                activation='relu',
                out_activation=None,
                **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        index = 0
        for units in arch[:-1]:
            self.layers.append(tf.keras.layers.Dense(units, activation=activation, name="{}_{}".format(kwargs['name'], index)))
            index+=1
        self.layers.append(tf.keras.layers.Dense(arch[-1], activation=out_activation, name="{}_{}".format(kwargs['name'], index)))

            
    def call(self, inputs, training=False):
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, self_interaction=False):
        super(SecondOrderFeatureInteraction, self).__init__()
        self.self_interaction = self_interaction

    def call(self, inputs, training = False):
        batch_size = tf.shape(inputs)[0]
        num_feas = tf.shape(inputs)[1]

        dot_products = tf.matmul(inputs, inputs, transpose_b=True)

        ones = tf.ones_like(dot_products, dtype=tf.float32)
        mask = tf.linalg.band_part(ones, 0, -1)
        out_dim = num_feas * (num_feas + 1) // 2

        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = num_feas * (num_feas - 1) // 2
        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
        return flat_interactions

class CrossLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            last_dim,
            use_bias: bool = True,
            **kwargs):

        super(CrossLayer, self).__init__(**kwargs)

        self._use_bias = use_bias

        self._dense = tf.keras.layers.Dense(
                last_dim,
                use_bias=self._use_bias,
        )

    @tf.function
    def call(self, x0: tf.Tensor, training=False) -> tf.Tensor:
        prod_output = self._dense(x0)
        return x0 * prod_output + x0
        # if x is None:
        #     x = x0
        # prod_output = self._dense(x)
        # return x0 * prod_output + x


class MLP_DCN(tf.keras.layers.Layer):
    def __init__(self,
                arch,
                activation='relu',
                out_activation=None,
                **kwargs):
        super(MLP_DCN, self).__init__(**kwargs)
        self.layers = []
        self.arch = arch
        self.final_act = out_activation
        index = 0
        for units in arch[:-1]:
            self.layers.append(tf.keras.layers.Dense(units, activation=activation, name="{}_{}".format(kwargs['name'], index)))
            index+=1
        self.layers.append(tf.keras.layers.Dense(arch[-1], activation=None, name="{}_{}".format(kwargs['name'], index)))
        if out_activation:
          # self.final_act = tf.keras.layers.Activation(out_activation, dtype='float32')
          self.final_act = tf.keras.layers.Activation(out_activation)

            
    def call(self, inputs, training=False):
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        if self.final_act:
            x = self.final_act(x)
        return x
