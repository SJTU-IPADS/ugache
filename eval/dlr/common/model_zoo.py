import tensorflow as tf
import hierarchical_parameter_server as hps
import sparse_operation_kit as sok

class InferenceModelHPS(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 dense_dim,
                 dense_model_path,
                 tf_key_type,
                 tf_vector_type,
                 **kwargs):
        super(InferenceModelHPS, self).__init__(**kwargs)
        
        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
        
        self.lookup_layer = hps.LookupLayer(model_name = "dlrm", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = self.tf_vector_type)
        self.dense_model = tf.keras.models.load_model(dense_model_path, compile=False)
        self.reshape_layer_final = tf.keras.layers.Reshape((), name = "reshape_final")
    
    def call(self, inputs):
        input_cat = inputs[0]
        input_dense = inputs[1]

        embeddings = tf.reshape(self.lookup_layer(input_cat),
                                shape=[-1, self.slot_num, self.embed_vec_size])
        logit = self.dense_model([embeddings, input_dense])
        return self.reshape_layer_final(logit)

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), sparse=False, dtype=self.tf_key_type),
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


class InferenceModelSOK(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 dense_dim,
                 dense_model_path,
                 tf_key_type,
                 tf_vector_type,
                 max_vocabulary_size_per_gpu,
                 **kwargs):
        super(InferenceModelSOK, self).__init__(**kwargs)
        
        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        
        self.lookup_layer = sok.All2AllDenseEmbedding(
                                                        max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                        embedding_vec_size=self.embed_vec_size,
                                                        slot_num=self.slot_num,
                                                        key_dtype=tf.uint32,
                                                        nnz_per_slot=1, use_hashtable=False)

        self.dense_model = tf.keras.models.load_model(dense_model_path, compile=False)
        self.reshape_layer_final = tf.keras.layers.Reshape((), name = "reshape_final")

    def call(self, inputs):
        input_cat = inputs[0]
        input_dense = inputs[1]

        embeddings = tf.reshape(self.lookup_layer(tf.cast(input_cat, tf.uint32)),
                                shape=[-1, self.slot_num, self.embed_vec_size])
        logit = self.dense_model([embeddings, input_dense])
        return self.reshape_layer_final(logit)

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), sparse=False, dtype=self.tf_key_type),
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


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

class DLRMHPS(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 embed_vec_size,
                 slot_num,
                 dense_dim,
                 arch_bot,
                 arch_top,
                 self_interaction,
                 tf_key_type,
                 tf_vector_type,
                 **kwargs):
        super(DLRMHPS, self).__init__(**kwargs)
        
        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
 
        self.lookup_layer = hps.LookupLayer(model_name = "dlrm", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = self.tf_vector_type)
        self.bot_nn = MLP(arch_bot, name = "bottom", out_activation='relu')
        self.top_nn = MLP(arch_top, name = "top", out_activation='sigmoid')
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)
        if self_interaction:
            self.interaction_out_dim = (self.slot_num+1) * (self.slot_num+2) // 2
        else:
            self.interaction_out_dim = self.slot_num * (self.slot_num+1) // 2
        self.reshape_layer0 = tf.keras.layers.Reshape((slot_num, arch_bot[-1]), name="reshape0")
        self.reshape_layer1 = tf.keras.layers.Reshape((1, arch_bot[-1]), name = "reshape1")
        self.reshape_layer_final = tf.keras.layers.Reshape((), name = "reshape_final")
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name = "concat1")
        self.concat2 = tf.keras.layers.Concatenate(axis=1, name = "concat2")
            
    def call(self, inputs, training=False):
        input_cat = inputs[0]
        input_dense = inputs[1]
        
        embedding_vector = self.lookup_layer(input_cat)
        embedding_vector = self.reshape_layer0(embedding_vector)
        input_dense = hps.hps_lib.nop_dep(dense=input_dense, emb = embedding_vector)
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

class DLRMSOK(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 embed_vec_size,
                 slot_num,
                 dense_dim,
                 arch_bot,
                 arch_top,
                 self_interaction,
                 tf_key_type,
                 tf_vector_type,
                 use_hashtable = False,
                 **kwargs):
        super(DLRMSOK, self).__init__(**kwargs)
        
        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu

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
        input_dense = sok.kit_lib.nop_dep(dense=input_dense, emb = embedding_vector)
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