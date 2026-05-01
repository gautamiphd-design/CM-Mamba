import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from einops.layers.tensorflow import Rearrange

# ---------------------------------------------------
# Channel + Spatial Gating
# ---------------------------------------------------
class ChannelSpatialGating(tf.keras.layers.Layer):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.fc1 = Dense(max(8, channels // 8), activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')

        self.conv = Conv1D(max(8, channels // 8), 3, padding='same', activation='relu')
        self.conv_out = Conv1D(1, 1, activation='sigmoid')

        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=None):
        # Channel attention
        ch = tf.reduce_mean(x, axis=1, keepdims=True)
        ch = self.fc1(ch)
        ch = self.fc2(ch)
        x = x * ch

        # Spatial attention
        sp = self.conv(x)
        sp = self.conv_out(sp)
        x = x * sp

        return self.dropout(x, training=training)


# ---------------------------------------------------
# Mamba Block with FULL SSM
# ---------------------------------------------------
class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, d_state=16, d_conv=4, expand=2, dropout_rate=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.d_inner = int(self.expand * self.d_model)

        # Layers
        self.norm = LayerNormalization()
        self.in_proj = Dense(self.d_inner * 2)
        self.conv1d = Conv1D(self.d_inner, self.d_conv, padding='same', groups=self.d_inner)

        self.dt_proj = Dense(self.d_inner)
        self.B_proj = Dense(self.d_state)
        self.C_proj = Dense(self.d_state)

        self.out_proj = Dense(self.d_model)

        # SSM parameters
        A = tf.range(1, self.d_state + 1, dtype=tf.float32)
        A = tf.tile(tf.expand_dims(A, 0), [self.d_inner, 1])

        self.A_log = self.add_weight(
            shape=(self.d_inner, self.d_state),
            initializer=tf.keras.initializers.Constant(tf.math.log(A)),
            trainable=True
        )

        self.D = self.add_weight(
            shape=(self.d_inner,),
            initializer="ones",
            trainable=True
        )

        # Gating
        self.gate = ChannelSpatialGating(self.d_inner, self.dropout_rate)

        # Residual scaling
        self.alpha = self.add_weight(
            shape=(),
            initializer="ones",
            trainable=True
        )

        super().build(input_shape)

    # ---------------------------------------------------
    # FULL SSM FUNCTION (core Mamba logic)
    # ---------------------------------------------------
    def ssm(self, x):
        A = -tf.exp(self.A_log)
        D = self.D

        delta = tf.nn.softplus(self.dt_proj(x))
        B = self.B_proj(x)
        C = self.C_proj(x)

        dA = tf.exp(tf.einsum('b l d, d n -> b l d n', delta, A))
        dB = tf.einsum('b l d, b l n -> b l d n', delta, B)

        dA_T = tf.transpose(dA, [1, 0, 2, 3])
        dB_T = tf.transpose(dB, [1, 0, 2, 3])

        h0 = tf.zeros([tf.shape(x)[0], self.d_inner, self.d_state])

        def step(h_prev, elems):
            a_t, b_t = elems
            return h_prev * a_t + b_t

        h = tf.scan(step, (dA_T, dB_T), initializer=h0)
        h = tf.transpose(h, [1, 0, 2, 3])

        y = tf.einsum('b l d n, b l n -> b l d', h, C) + x * tf.expand_dims(D, 0)

        return y

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    def call(self, x, training=None):
        x_res = x
        x = self.norm(x)

        x_proj = self.in_proj(x)
        x_feat, x_gate = tf.split(x_proj, 2, axis=-1)

        x_conv = tf.nn.silu(self.conv1d(x_feat))

        # Apply gating
        x_gated = self.gate(x_conv, training=training)

        # Apply SSM
        y_ssm = self.ssm(x_gated)

        # Final gating
        y = y_ssm * tf.nn.silu(x_gate)
        y = self.out_proj(y)

        return x_res + self.alpha * y


# ---------------------------------------------------
# Final Model
# ---------------------------------------------------
def create_model(input_shape=(224,224,3), num_classes=3):

    inputs = Input(shape=input_shape)

    # Data augmentation
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)

    # Patch embedding
    x = Conv2D(64, (16,16), strides=16)(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)

    # Sequence conversion
    x = Rearrange('b h w c -> b (h w) c')(x)

    # Mamba blocks
    for _ in range(4):
        x = MambaBlock()(x)

    # Classifier
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='gelu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
    Model.summary() 
