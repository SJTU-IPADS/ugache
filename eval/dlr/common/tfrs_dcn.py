from typing import cast, Dict, Optional, Sequence, Tuple, Union

import tensorflow as tf
import hierarchical_parameter_server as hps
import sparse_operation_kit as sok

# from tensorflow_recommenders.layers import feature_interaction as feature_interaction_lib


from typing import Union, Text, Optional
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


class MLP(tf.keras.layers.Layer):
    def __init__(self,
                arch,
                activation='relu',
                out_activation=None,
                **kwargs):
        super(MLP, self).__init__(**kwargs)
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


class DCNSOK(tf.keras.models.Model):
  """A configurable ranking model.
  This class represents a sensible and reasonably flexible configuration for a
  ranking model that can be used for tasks such as CTR prediction.
  It can be customized as needed, and its constituent blocks can be changed by
  passing user-defined alternatives.
  For example:
  - Pass
    `feature_interaction = tfrs.layers.feature_interaction.DotInteraction()`
    to train a DLRM model, or pass
    ```
    feature_interaction = tf.keras.Sequential([
      tf.keras.layers.Concatenate(),
      tfrs.layers.feature_interaction.Cross()
    ])
    ```
    to train a DCN model.
  - Pass `task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())`
    to train a CTR prediction model, and
    `tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())` to train
    a rating prediction model.
  Changing these should cover a broad range of models, but this class is not
  intended to cover all possible use cases.  For full flexibility, inherit
  from `tfrs.models.Model` and provide your own implementations of
  the `compute_loss` and `call` methods.
  """

  def __init__(
      self,
      max_vocabulary_size_per_gpu,
      embed_vec_size,
      slot_num,
      dense_dim,
      use_hashtable=False,
      bottom_stack: Optional[tf.keras.layers.Layer] = None,
      feature_interaction: Optional[tf.keras.layers.Layer] = None,
      top_stack: Optional[tf.keras.layers.Layer] = None) -> None:
    """Initializes the model.
    Args:
      embedding_layer: The embedding layer is applied to categorical features.
        It expects a string-to-tensor (or SparseTensor/RaggedTensor) dict as
        an input, and outputs a dictionary of string-to-tensor of feature_name,
        embedded_value pairs.
        {feature_name_i: tensor_i} -> {feature_name_i: emb(tensor_i)}.
      bottom_stack: The `bottom_stack` layer is applied to dense features before
        feature interaction. If None, an MLP with layer sizes [256, 64, 16] is
        used. For DLRM model, the output of bottom_stack should be of shape
        (batch_size, embedding dimension).
      feature_interaction: Feature interaction layer is applied to the
        `bottom_stack` output and sparse feature embeddings. If it is None,
        DotInteraction layer is used.
      top_stack: The `top_stack` layer is applied to the `feature_interaction`
        output. The output of top_stack should be in the range [0, 1]. If it is
        None, MLP with layer sizes [512, 256, 1] is used.
      task: The task which the model should optimize for. Defaults to a
        `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable
        for tasks like click prediction.
    """

    super().__init__()
    self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
    self.embed_vec_size = embed_vec_size
    self.slot_num = slot_num
    self.dense_dim = dense_dim

    self._embedding_layer = sok.All2AllDenseEmbedding(
                                                    max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                    embedding_vec_size=self.embed_vec_size,
                                                    slot_num=self.slot_num,
                                                    key_dtype=tf.uint32,
                                                    nnz_per_slot=1, use_hashtable=use_hashtable)

    self.reshape_layer1  = tf.keras.layers.Reshape((slot_num, 1), name = "reshape1")
    self.reshape_layer2  = tf.keras.layers.Reshape((self.slot_num * self.embed_vec_size,), name = "reshape2")
    self.reshape_layer_final  = tf.keras.layers.Reshape((), name = "reshape_final")
    self._bottom_stack = bottom_stack if bottom_stack else MLP(arch=[256, 64, 16], out_activation="relu", name="bottom")
    self._top_stack = top_stack if top_stack else MLP(arch=[256, 256, 1], out_activation="sigmoid", name="top")
    self._cross_dense = tf.keras.layers.Dense(
            self.embed_vec_size * self.slot_num + self._bottom_stack.arch[-1], name="cross"
    )
    # self._feature_interaction = (feature_interaction if feature_interaction
    #                              else feature_interaction_lib.DotInteraction())
    # self._feature_interaction = tf.keras.Sequential([
    #   tf.keras.layers.Concatenate(),
    #   CrossLayer(self.embed_vec_size * self.slot_num + self._bottom_stack.arch[-1])
    # ])

  @tf.function(jit_compile=True) 
  def __non_lookup(self, sparse_embeddings, input_dense):
    sparse_embeddings = self.reshape_layer2(sparse_embeddings)
    # sparse_embeddings = tf.ones([input_cat.shape[0], self.slot_num * self.embed_vec_size])
    # (batch_size, emb).
    dense_embedding_vec = self._bottom_stack(input_dense)

    interaction_output = tf.concat([tf.cast(sparse_embeddings, tf.float16), dense_embedding_vec], axis=1)
    
    # interaction_output = interaction_output * tf.cast(self._cross_dense(interaction_output), tf.float32) + interaction_output
    interaction_output = interaction_output * self._cross_dense(interaction_output) + interaction_output

    # interaction_output = self._feature_interaction([sparse_embeddings, dense_embedding_vec])
    feature_interaction_output = tf.concat(
        [dense_embedding_vec, interaction_output], axis=1)

    prediction = self._top_stack(feature_interaction_output)

    return self.reshape_layer_final(prediction)

  @tf.function
  def call(self, inputs, training=False):
    """Executes forward and backward pass, returns loss.
    Args:
      inputs: Model function inputs (features and labels).
    Returns:
      loss: Scalar tensor.
    """
    input_cat = inputs[0]
    input_dense = inputs[1]

    input_cat = self.reshape_layer1(input_cat)
    sparse_embeddings = self._embedding_layer(input_cat, training)
    return self.__non_lookup(sparse_embeddings, input_dense)


class DCNHPS(tf.keras.models.Model):
  """A configurable ranking model.
  This class represents a sensible and reasonably flexible configuration for a
  ranking model that can be used for tasks such as CTR prediction.
  It can be customized as needed, and its constituent blocks can be changed by
  passing user-defined alternatives.
  For example:
  - Pass
    `feature_interaction = tfrs.layers.feature_interaction.DotInteraction()`
    to train a DLRM model, or pass
    ```
    feature_interaction = tf.keras.Sequential([
      tf.keras.layers.Concatenate(),
      tfrs.layers.feature_interaction.Cross()
    ])
    ```
    to train a DCN model.
  - Pass `task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())`
    to train a CTR prediction model, and
    `tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())` to train
    a rating prediction model.
  Changing these should cover a broad range of models, but this class is not
  intended to cover all possible use cases.  For full flexibility, inherit
  from `tfrs.models.Model` and provide your own implementations of
  the `compute_loss` and `call` methods.
  """

  def __init__(
      self,
      embed_vec_size,
      slot_num,
      dense_dim,
      bottom_stack: Optional[tf.keras.layers.Layer] = None,
      feature_interaction: Optional[tf.keras.layers.Layer] = None,
      top_stack: Optional[tf.keras.layers.Layer] = None) -> None:
    """Initializes the model.
    Args:
      embedding_layer: The embedding layer is applied to categorical features.
        It expects a string-to-tensor (or SparseTensor/RaggedTensor) dict as
        an input, and outputs a dictionary of string-to-tensor of feature_name,
        embedded_value pairs.
        {feature_name_i: tensor_i} -> {feature_name_i: emb(tensor_i)}.
      bottom_stack: The `bottom_stack` layer is applied to dense features before
        feature interaction. If None, an MLP with layer sizes [256, 64, 16] is
        used. For DLRM model, the output of bottom_stack should be of shape
        (batch_size, embedding dimension).
      feature_interaction: Feature interaction layer is applied to the
        `bottom_stack` output and sparse feature embeddings. If it is None,
        DotInteraction layer is used.
      top_stack: The `top_stack` layer is applied to the `feature_interaction`
        output. The output of top_stack should be in the range [0, 1]. If it is
        None, MLP with layer sizes [512, 256, 1] is used.
      task: The task which the model should optimize for. Defaults to a
        `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable
        for tasks like click prediction.
    """

    super().__init__()
    self.slot_num = slot_num
    self.dense_dim = dense_dim
    self.embed_vec_size = embed_vec_size

    self._embedding_layer = hps.LookupLayer(model_name = "dcn", 
                                        table_id = 0,
                                        emb_vec_size = self.embed_vec_size,
                                        emb_vec_dtype = tf.float32)

    self.reshape_layer2  = tf.keras.layers.Reshape((self.slot_num * self.embed_vec_size,), name = "reshape2")
    self.reshape_layer_final  = tf.keras.layers.Reshape((), name = "reshape_final")
    self._bottom_stack = bottom_stack if bottom_stack else MLP(arch=[256, 64, 16], out_activation="relu", name="bottom")
    self._top_stack = top_stack if top_stack else MLP(arch=[256, 256, 1], out_activation="sigmoid", name="top")
    self._cross_dense = tf.keras.layers.Dense(
            self.embed_vec_size * self.slot_num + self._bottom_stack.arch[-1], name="cross"
    )
    # self._feature_interaction = (feature_interaction if feature_interaction
    #                              else feature_interaction_lib.DotInteraction())
    # self._feature_interaction = tf.keras.Sequential([
    #   tf.keras.layers.Concatenate(),
    #   CrossLayer(self.embed_vec_size * self.slot_num + self._bottom_stack.arch[-1])
    # ])

  @tf.function(jit_compile=True) 
  def __non_lookup(self, sparse_embeddings, input_dense):
    sparse_embeddings = self.reshape_layer2(sparse_embeddings)
    # sparse_embeddings = tf.ones([input_cat.shape[0], self.slot_num * self.embed_vec_size])
    # (batch_size, emb).
    dense_embedding_vec = self._bottom_stack(input_dense)

    interaction_output = tf.concat([tf.cast(sparse_embeddings, tf.float16), dense_embedding_vec], axis=1)
    
    # interaction_output = interaction_output * tf.cast(self._cross_dense(interaction_output), tf.float32) + interaction_output
    interaction_output = interaction_output * self._cross_dense(interaction_output) + interaction_output

    # interaction_output = self._feature_interaction([sparse_embeddings, dense_embedding_vec])
    feature_interaction_output = tf.concat(
        [dense_embedding_vec, interaction_output], axis=1)

    prediction = self._top_stack(feature_interaction_output)

    return self.reshape_layer_final(prediction)

  @tf.function
  def call(self, inputs):
    """Executes forward and backward pass, returns loss.
    Args:
      inputs: Model function inputs (features and labels).
    Returns:
      loss: Scalar tensor.
    """
    input_cat = inputs[0]
    input_dense = inputs[1]

    sparse_embeddings = self._embedding_layer(input_cat)
    input_dense = hps.hps_lib.nop_dep(dense=input_dense, emb = sparse_embeddings)

    return self.__non_lookup(sparse_embeddings, input_dense)
