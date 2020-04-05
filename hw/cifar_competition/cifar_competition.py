#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Training
  parser.add_argument("--batch_size", default=45, type=int, help="Batch size.")
  parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
  # Hidden layers
  parser.add_argument("--hidden_layer_size", default=160, type=int, help="Size of the hidden layer.")
  parser.add_argument("--hidden_layers", default=7, type=int, help="Number of hidden layers")
  parser.add_argument("--activation", default="relu", type=str, help="Activation function")
  # Seed
  parser.add_argument("--seed", default=42, type=int, help="Random seed.")
  # Performance
  parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
  args = parser.parse_args([] if "__file__" not in globals() else None)

  # Select the correct activation function
  if (args.activation == "relu"):
    activation = tf.nn.relu
  elif (args.activation == "tanh"):
    activation = tf.nn.tanh
  elif (args.activation == "sigmoid"):
    activation = tf.nn.sigmoid
  else:
    activation = None

  # Fix random seeds and threads
  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  tf.config.threading.set_inter_op_parallelism_threads(args.threads)
  tf.config.threading.set_intra_op_parallelism_threads(args.threads)

  # Report only errors by default
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  # Create logdir name
  args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(globals().get("__file__", "notebook")),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
  ))

  # Load data
  cifar = CIFAR10()

  # Create the model
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer([CIFAR10.H, CIFAR10.W, CIFAR10.C]))
  model.add(tf.keras.layers.Flatten())
  [model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation=activation)) for _ in range(args.hidden_layers)]
  model.add(tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax))

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
    cifar.train.data["images"], cifar.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
    callbacks=[tb_callback],
  )

  # Evaluate the model
  test_logs = model.evaluate(
    cifar.dev.data["images"], cifar.dev.data["labels"], batch_size=args.batch_size,
  )

  # Generate test set annotations, but in args.logdir to allow parallel execution.
  with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
    for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
      print(np.argmax(probs), file=out_file)
