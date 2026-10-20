"""Red G generadora (docs/models.md §G).

Entrada: (10,3) Lab normalizado + máscara multi-hot (10,) + embedding armonía 4→8 (opcional).
Arquitectura: aplanado 10*3+10+8=48 → MLP 48→128→128 ReLU dropout0.1
 → cabeza paleta (15 = 5 colores *3) + cabeza compatibilidad (1 sigmoide).

Entrada multi-input para condicionar por armonía; en inferencia si no hay
armonía se pasa vector cero.

TensorFlow importado a nivel superior (solo se carga en .venv-tf y en
inference de forma perezosa).
"""
from __future__ import annotations

import keras
import tensorflow as tf

HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]
HARMONY_TO_IDX = {h: i for i, h in enumerate(HARMONY_LABELS)}


def build_generate_model() -> keras.Model:
    palette_in = keras.layers.Input(shape=(10, 3), name="palette_lab")
    mask_in = keras.layers.Input(shape=(10,), name="mask")
    harmony_in = keras.layers.Input(shape=(4,), name="harmony")  # one-hot 4

    # embedding 4 -> 8
    harmony_emb = keras.layers.Dense(8, activation="relu", name="harmony_emb")(harmony_in)

    flat_pal = keras.layers.Flatten(name="flat_pal")(palette_in)  # 30
    concat = keras.layers.Concatenate(name="concat")([flat_pal, mask_in, harmony_emb])  # 48

    x = keras.layers.Dense(128, activation="relu", name="dense_1")(concat)
    x = keras.layers.Dropout(0.1, name="dropout_1")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_2")(x)
    x = keras.layers.Dropout(0.1, name="dropout_2")(x)

    palette_out = keras.layers.Dense(15, activation="linear", name="palette")(x)
    score_out = keras.layers.Dense(1, activation="sigmoid", name="score")(x)

    model = keras.Model(
        inputs=[palette_in, mask_in, harmony_in],
        outputs=[palette_out, score_out],
        name="generate",
    )
    return model


@keras.saving.register_keras_serializable(package="combina2")
def generate_palette_loss(y_true, y_pred):
    """MSE + 0.1*CIEDE2000 aprox (registrable para .keras)."""
    mu = 0.1
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    def denorm(t):
        r = tf.reshape(t, (-1, 5, 3))
        l = r[..., 0:1] * 100.0
        ab = r[..., 1:3] * 128.0
        return tf.concat([l, ab], axis=-1)

    pred_lab = denorm(y_pred)
    true_lab = denorm(y_true)
    diff = pred_lab - true_lab
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    ci_el = tf.reduce_mean(dist)
    return mse + mu * ci_el


def palette_loss_with_ciede(mu: float = 0.1):
    """Compatibilidad: wrapper que ignora mu y devuelve loss registrada si mu==0.1."""
    if abs(mu - 0.1) < 1e-6:
        return generate_palette_loss
    # fallback no serializable (solo entrenamiento efímero); para guardar usar generate_palette_loss
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        def denorm(t):
            r = tf.reshape(t, (-1, 5, 3))
            l = r[..., 0:1] * 100.0
            ab = r[..., 1:3] * 128.0
            return tf.concat([l, ab], axis=-1)

        pred_lab = denorm(y_pred)
        true_lab = denorm(y_true)
        diff = pred_lab - true_lab
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
        ci_el = tf.reduce_mean(dist)
        return mse + mu * ci_el

    return loss


GENERATE_CUSTOM_OBJECTS = {"generate_palette_loss": generate_palette_loss}
