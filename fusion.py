import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization

base_cxr_model = create_medmamba_model(model_name="base_cxr")
base_ct_model  = create_medmamba_model(model_name="base_ct")

dummy = tf.random.normal([1,224,224,3])
base_cxr_model(dummy)
base_ct_model(dummy)

base_cxr_model.load_weights(cxr_weights_path)
base_ct_model.load_weights(ct_weights_path)

cxr_encoder = Model(
    inputs=base_cxr_model.input,
    outputs=base_cxr_model.get_layer("base_cxr_gap").output,
    name="cxr_encoder"
)

ct_encoder = Model(
    inputs=base_ct_model.input,
    outputs=base_ct_model.get_layer("base_ct_gap").output,
    name="ct_encoder"
)


cxr_encoder.trainable = False
ct_encoder.trainable  = False

print(" Encoders ready")


cxr_input = Input(shape=(224,224,3), name="cxr_input")
ct_input  = Input(shape=(224,224,3), name="ct_input")

# ---- Step 1: Feature Extraction ----
cxr_feat = cxr_encoder(cxr_input)   # (B, D)
ct_feat  = ct_encoder(ct_input)     # (B, D)

# ---- Step 2: Feature Harmonization (MAFH) ----
harm_feat = MAFH(units=128)([ct_feat, cxr_feat])   # aligned representation

# ---- Step 3: Cross-Modal Fusion (MECAF) ----
fusion_feat = MECAF()([harm_feat, harm_feat])

# ---- Step 4: Classification ----
x = Dense(128, activation='gelu')(fusion_feat)
x = LayerNormalization()(x)
output = Dense(3, activation='softmax')(x)

fusion_model = Model(inputs=[cxr_input, ct_input], outputs=output)

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fusion_model.summary()
