#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
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

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
    # For `SGD`, `args.momentum` can be specified.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `learning_rate` argument.
    # - If `args.decay` is set, then
    #   - for `polynomial`, use `tf.optimizers.schedules.PolynomialDecay`
    #     using the given `args.learning_rate_final`;
    #   - for `exponential`, use `tf.optimizers.schedules.ExponentialDecay`
    #     and set `decay_rate` appropriately to reach `args.learning_rate_final`
    #     just after the training (and keep the default `staircase=False`).
    #   In both cases, `decay_steps` should be total number of training batches
    #   and you should pass the created `{Polynomial,Exponential}Decay` to
    #   the optizer using the `learning_rate` constructor argument.
    #   The size of the training MNIST dataset it `mnist.train.size` and you
    #   can assume is it divisible by `args.batch_size`.
    #
    #   If a learning rate schedule is used, you can find out the current learning
    #   rate by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
    #   so after training this value should be `args.learning_rate_final`.
    
    decay_steps = (mnist.train.size / args.batch_size) * args.epochs

    if (args.decay is None):
      learning_rate = args.learning_rate
    elif (args.decay == "polynomial"):
      learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=decay_steps,
        end_learning_rate=args.learning_rate_final
      )
    elif (args.decay == "exponential"):
      learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=decay_steps,
        decay_rate=args.learning_rate_final / args.learning_rate
      )

    if (args.optimizer == "SGD"):
      momentum = 0.0 if args.momentum is None else args.momentum
      optimizer = tf.keras.optimizers.SGD(learning_rate, momentum)
    elif (args.optimizer == "Adam"):
      optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )
    
    if (args.decay is not None):
      print(model.optimizer.learning_rate(model.optimizer.iterations))

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
    )

    tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})

    # TODO: Write test accuracy as percentages rounded to two decimal places.
    with open("mnist_training.out", "w") as out_file:
        print("{:.2f}".format(100 * test_logs[1]), file=out_file)
