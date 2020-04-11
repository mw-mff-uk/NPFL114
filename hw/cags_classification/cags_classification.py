#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from math import ceil

from cags_dataset import CAGS
import efficient_net

if __name__ == "__main__":
    # section [SETTINGS]
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--batch_size", default=42, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    # CNN Layers
    parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
    # Image data generator
    parser.add_argument("--rotation_range", default=10, type=int)
    parser.add_argument("--zoom_range", default=0.2, type=int)
    parser.add_argument("--width_shift_range", default=0.1, type=int)
    parser.add_argument("--height_shift_range", default=0.1, type=int)
    parser.add_argument("--horizontal_flip", default=1, type=int)
    # Seed
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    # Performance
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    # Misc
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Create logdir if not exists
    if (not os.path.isdir(args.logdir)):
        os.mkdir(args.logdir)

    # section [MODEL]
    # Load the data
    cags = CAGS()

    train = cags.train.map(CAGS.parse).shuffle(CAGS.TRAIN_EXAMPLES, seed=args.seed).batch(args.batch_size)
    dev = cags.dev.map(CAGS.parse).batch(args.batch_size)
    test = cags.test.map(CAGS.parse).batch(1)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    # Build the model
    hidden = efficientnet_b0.output[0]
    # [...]
    outputs = layers.Dense(len(CAGS.LABELS), activation="softmax")(hidden)

    model = tf.keras.models.Model(inputs=efficientnet_b0.input, outputs=outputs)

    # Make all layers non trainable (except BatchNormalization)
    for layer in efficientnet_b0.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Tensor board
    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # Fit the model
    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        # steps_per_epoch=ceil(CAGS.TRAIN_EXAMPLES / args.batch_size),
        # validation_steps=ceil(CAGS.TRAIN_EXAMPLES / args.batch_size),
        callbacks=[tb_callback]
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("cags_classification_latest.txt", "w", encoding="utf-8") as out_file_latest:
        with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
            test_probabilities = model.predict(test)
            for probs in test_probabilities:
                print(np.argmax(probs), file=out_file)
                print(np.argmax(probs), file=out_file_latest)
