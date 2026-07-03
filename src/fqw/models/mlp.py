"""Deep MLP architecture."""

from tensorflow.keras import layers, models, optimizers, regularizers


def build_mlp_model(
    input_shape: tuple[int, int],
    learning_rate: float = 1e-3,
    dropout: float = 0.7,
    l2_decay: float = 1e-5,
):
    """6-layer MLP from thesis experiments."""
    model = models.Sequential(
        [
            layers.Flatten(input_shape=input_shape, name="flatten_mlp"),
            layers.Dense(300, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(200, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(100, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(50, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(25, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(10, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_decay)),
            layers.Dropout(dropout),
            layers.Dense(3, activation="softmax", dtype="float32", name="output_mlp"),
        ]
    )
    try:
        opt = optimizers.legacy.Adam(learning_rate=learning_rate)
    except AttributeError:
        opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
