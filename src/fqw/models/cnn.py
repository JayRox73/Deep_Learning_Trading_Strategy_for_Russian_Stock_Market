"""Neural network architectures."""

from tensorflow.keras import layers, models, optimizers


def build_cnn_model(
    input_shape: tuple[int, int],
    learning_rate: float = 1e-4,
    dropout: float = 0.3,
):
    """Regular 1D CNN used in the main thesis pipeline."""
    model = models.Sequential(
        [
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(3, activation="softmax"),
        ]
    )

    try:
        opt = optimizers.legacy.Adam(learning_rate=learning_rate)
    except AttributeError:
        opt = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
