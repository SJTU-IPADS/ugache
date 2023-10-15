from .common import tf, MLP, SecondOrderFeatureInteraction, Optional, MLP_DCN
import sparse_operation_kit as sok

class DLRM(tf.keras.models.Model):
    def __init__(self,
                 embed_vec_size,
                 slot_num,
                 dense_dim,
                 arch_bot = None,
                 arch_top = None,
                 self_interaction = False,
                 tf_key_type = tf.uint32,
                 tf_vector_type = tf.float32,
                 max_vocabulary_size_per_gpu = None,
                 use_hashtable = False,
                 **kwargs):
        super(DLRM, self).__init__(**kwargs)
        
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
        assert(max_vocabulary_size_per_gpu != None)
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        if arch_bot is None:
          arch_bot = [256, 128, embed_vec_size]
        if arch_top is None:
          arch_top = [256, 128, 1]

        self.lookup_layer = sok.All2AllDenseEmbedding(
                                                        max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                        embedding_vec_size=self.embed_vec_size,
                                                        slot_num=self.slot_num,
                                                        key_dtype=tf.uint32,
                                                        trainable=False,
                                                        nnz_per_slot=1, use_hashtable=use_hashtable)

        self.bot_nn = MLP(arch_bot, name = "bottom", out_activation='relu')
        self.top_nn = MLP(arch_top, name = "top", out_activation='sigmoid')
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)
        if self_interaction:
            self.interaction_out_dim = (self.slot_num+1) * (self.slot_num+2) // 2
        else:
            self.interaction_out_dim = self.slot_num * (self.slot_num+1) // 2
        self.reshape_layer  = tf.keras.layers.Reshape((slot_num, 1), name = "reshape")
        self.reshape_layer0 = tf.keras.layers.Reshape((slot_num, arch_bot[-1]), name="reshape0")
        self.reshape_layer1 = tf.keras.layers.Reshape((1, arch_bot[-1]), name = "reshape1")
        self.reshape_layer_final = tf.keras.layers.Reshape((), name = "reshape_final")
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name = "concat1")
        self.concat2 = tf.keras.layers.Concatenate(axis=1, name = "concat2")
            
    def call(self, inputs, training=False):
        input_cat = inputs[0]
        input_dense = inputs[1]
        
        input_cat = self.reshape_layer(input_cat)
        embedding_vector = self.lookup_layer(input_cat, training)
        embedding_vector = self.reshape_layer0(embedding_vector)
        dense_x = self.bot_nn(input_dense)
        concat_features = self.concat1([embedding_vector, self.reshape_layer1(dense_x)])
        
        Z = self.interaction_op(concat_features)
        z = self.concat2([dense_x, Z])
        logit = self.top_nn(z)
        return self.reshape_layer_final(logit)

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), sparse=False, dtype=self.tf_key_type), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()
    

class DCN(tf.keras.models.Model):
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
      max_vocabulary_size_per_gpu = None,
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
        feature interaction. If None, an MLP_DCN with layer sizes [256, 64, 16] is
        used. For DLRM model, the output of bottom_stack should be of shape
        (batch_size, embedding dimension).
      feature_interaction: Feature interaction layer is applied to the
        `bottom_stack` output and sparse feature embeddings. If it is None,
        DotInteraction layer is used.
      top_stack: The `top_stack` layer is applied to the `feature_interaction`
        output. The output of top_stack should be in the range [0, 1]. If it is
        None, MLP_DCN with layer sizes [512, 256, 1] is used.
      task: The task which the model should optimize for. Defaults to a
        `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable
        for tasks like click prediction.
    """

    super().__init__()
    assert(max_vocabulary_size_per_gpu != None)
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
    self._bottom_stack = bottom_stack if bottom_stack else MLP_DCN(arch=[256, 64, 16], out_activation="relu", name="bottom")
    self._top_stack = top_stack if top_stack else MLP_DCN(arch=[256, 256, 1], out_activation="sigmoid", name="top")
    self._cross_dense = tf.keras.layers.Dense(
            self.embed_vec_size * self.slot_num + self._bottom_stack.arch[-1], name="cross"
    )

  @tf.function(jit_compile=True) 
  def __non_lookup(self, sparse_embeddings, input_dense):
    sparse_embeddings = self.reshape_layer2(sparse_embeddings)
    dense_embedding_vec = self._bottom_stack(input_dense)

    interaction_output = tf.concat([tf.cast(sparse_embeddings, tf.float16), dense_embedding_vec], axis=1)
    
    interaction_output = interaction_output * self._cross_dense(interaction_output) + interaction_output

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
