"""CDT 1D CNN architecture."""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cdt_1d_cnn(
    n_features: int,
    window_size: int = 24,
    n_classes: int = 3,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    l2_decay: float = 1e-5,
):
    inp = layers.Input(shape=(window_size, n_features))
    x = layers.Conv1D(32, 4, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_decay))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4, strides=4)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(3, strides=3)(x)
    x = layers.Conv1D(128, 2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, strides=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_decay))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out, name="CDT_1D_CNN_Fast")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
