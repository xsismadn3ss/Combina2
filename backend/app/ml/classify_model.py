"""Red C clasificadora (docs/models.md §C).

Entradas soportadas:
- 54 dims (`classify-v2`): features de conjunto ordenadas por tono
  (`app.ml.color_features.palette_features`).
- 40 dims (`classify-v1`, rollback): 10*Lab normalizado + 10 pesos de E.

Arquitectura: MLP input_dim→64→32 ReLU → softmax 4
Loss: cross-entropy + label smoothing 0.05
"""
from __future__ import annotations

import keras

HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]
HARMONY_TO_IDX = {h: i for i, h in enumerate(HARMONY_LABELS)}
IDX_TO_HARMONY = {i: h for h, i in HARMONY_TO_IDX.items()}


def build_classify_model(input_dim: int = 54) -> keras.Model:
    inp = keras.layers.Input(shape=(int(input_dim),), name="palette_feat")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inp)
    x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    out = keras.layers.Dense(4, activation="softmax", name="harmony")(x)
    model = keras.Model(inputs=inp, outputs=out, name="classify")
    return model


def cce_label_smoothing(y_true, y_pred):
    # y_true is one-hot (B,4), y_pred softmax
    # use keras loss with label_smoothing 0.05
    import tensorflow as tf

    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=0.05
    )
