#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cifar = CIFAR10()

    # Create the model
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = tf.keras.layers.Conv2D(16, 3, 2, "same", activation=tf.nn.relu)(inputs)
    hidden = tf.keras.layers.Conv2D(16, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

    # Train the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # TODO: Create data augmenting `tf.keras.preprocessing.image.ImageDataGenerator`.
    # Specify:
    # - rotation range of 20 degrees,
    # - zoom range of 0.2 (20%),
    # - width shift range and height shift range of 0.1 (10%),
    # - allow horizontal flips
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # TODO: Train using the generator. To augment data, use
    # `train_generator.flow` and specify:
    # - first 5000 of cifar.train.data["images"] as inputs
    # - first 5000 of cifar.train.data["labels"] as target
    # - batch_size of args.batch_size
    # - args.seed as random seed
    model.fit(
        train_generator.flow(
            cifar.train.data["images"][:5000],
            cifar.train.data["labels"][:5000],
            batch_size=args.batch_size,
            seed=args.seed
        ),
        shuffle=False, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Evaluate and print dev set accuracy
    dev_logs = model.evaluate(cifar.dev.data["images"], cifar.dev.data["labels"], batch_size=args.batch_size)
    accuracy = dev_logs[model.metrics_names.index("accuracy")]

    with open("image_augmentation.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
