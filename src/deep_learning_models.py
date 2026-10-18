"""TensorFlow/Keras models and tf.data input pipelines for tabular data."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .utils import SEED

N_CLASSES = 3


def make_tf_dataset(X: np.ndarray, y: np.ndarray | None = None, batch_size: int = 512,
                    shuffle: bool = False, sample_weight: np.ndarray | None = None,
                    seed: int = SEED, cache: bool = True) -> tf.data.Dataset:
    """Build a tf.data pipeline from already-preprocessed arrays.

    Only the training pipeline is shuffled (reshuffled every epoch); validation
    and test pipelines keep their order so predictions align with labels.
    Preprocessing is fitted beforehand on the training split, so no
    validation/test statistics enter the training pipeline.
    """
    X = np.asarray(X, dtype=np.float32)
    if y is None:
        tensors = X
    elif sample_weight is None:
        tensors = (X, np.asarray(y, dtype=np.int32))
    else:
        tensors = (X, np.asarray(y, dtype=np.int32), np.asarray(sample_weight, dtype=np.float32))
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 100_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_sequential_baseline(input_dim: int, hidden_units: int = 32,
                              learning_rate: float = 1e-3) -> keras.Model:
    """Baseline MLP (Sequential API): one ReLU hidden layer and a softmax output."""
    model = keras.Sequential([
        keras.Input(shape=(input_dim,), name="features"),
        layers.Dense(hidden_units, activation="relu", name="hidden_1"),
        layers.Dense(N_CLASSES, activation="softmax", name="class_probabilities"),
    ], name="sequential_baseline")
    return _compile(model, learning_rate)


def build_sequential_regularized(input_dim: int, hidden_units: tuple[int, ...] = (128, 64, 32),
                                 dropout: float = 0.3, l2: float = 1e-4,
                                 learning_rate: float = 1e-3, batch_norm: bool = True) -> keras.Model:
    """Deeper Sequential MLP with batch normalisation, dropout and L2 weight decay."""
    model = keras.Sequential(name="sequential_regularized")
    model.add(keras.Input(shape=(input_dim,), name="features"))
    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2), name=f"dense_{i}"))
        if batch_norm:
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Activation("relu", name=f"relu_{i}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, seed=SEED + i, name=f"dropout_{i}"))
    model.add(layers.Dense(N_CLASSES, activation="softmax", name="class_probabilities"))
    return _compile(model, learning_rate)


def build_wide_and_deep(input_dim: int, deep_units: tuple[int, ...] = (128, 64),
                        dropout: float = 0.3, l2: float = 1e-4,
                        learning_rate: float = 1e-3) -> keras.Model:
    """Wide-and-deep network (Functional API).

    A linear 'wide' path (equivalent to multinomial logistic regression) and a
    non-linear 'deep' path read the same inputs; their logits are added before
    the softmax. The wide path preserves the strong additive signal that
    logistic regression captures, while the deep path models interactions.
    This two-branch graph cannot be expressed with the Sequential API.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    wide_logits = layers.Dense(N_CLASSES, kernel_regularizer=regularizers.l2(l2), name="wide_logits")(inputs)
    x = inputs
    for i, units in enumerate(deep_units, start=1):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2), name=f"deep_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"deep_bn_{i}")(x)
        x = layers.Activation("relu", name=f"deep_relu_{i}")(x)
        x = layers.Dropout(dropout, seed=SEED + i, name=f"deep_dropout_{i}")(x)
    deep_logits = layers.Dense(N_CLASSES, name="deep_logits")(x)
    logits = layers.Add(name="combined_logits")([wide_logits, deep_logits])
    outputs = layers.Activation("softmax", name="class_probabilities")(logits)
    model = keras.Model(inputs, outputs, name="functional_wide_and_deep")
    return _compile(model, learning_rate)


def train_keras_model(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset,
                      run_dir: str | Path, epochs: int = 60, patience: int = 6,
                      class_weight: dict | None = None, verbose: int = 0):
    """Train with early stopping, best-model checkpointing, TensorBoard and CSV logging.

    Returns ``(model, history_dict, train_seconds, epochs_run)``; the returned
    model holds the weights of the best validation-loss epoch.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard"), histogram_freq=0),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
                        class_weight=class_weight, verbose=verbose)
    seconds = time.perf_counter() - start
    return model, history.history, seconds, len(history.history["loss"])


def balanced_class_weights(y: np.ndarray, power: float = 1.0) -> dict:
    """Inverse-frequency class weights, n / (k * n_c), optionally softened by ``power``."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = (len(y) / (len(classes) * counts)) ** power
    return {int(c): float(v) for c, v in zip(classes, w)}


def predict_proba(model: keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    return model.predict(ds, verbose=0)
