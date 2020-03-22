import argparse
import datetime
import os
import re

from uppercase_data import UppercaseData

import numpy as np
import tensorflow as tf


if (__name__ == "__main__"):
  parser = argparse.ArgumentParser()
  # Dataset
  parser.add_argument("--alphabet_window", default=4, type=int, help="Radius of the window in Alphabet dataset")
  parser.add_argument("--alphabet_size", default=80, type=int, help="The number of symbols in the Alphabet dataset")
  # Training shape
  parser.add_argument("--batch_size", default=2000, type=int, help="Batch size.")
  parser.add_argument("--epochs", default=60, type=int, help="Number of epochs.")
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

  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  tf.config.threading.set_inter_op_parallelism_threads(args.threads)
  tf.config.threading.set_intra_op_parallelism_threads(args.threads)

  # Create logdir name
  args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(globals().get("__file__", "notebook")),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
  ))

  # Read data from zip file
  data = UppercaseData(args.alphabet_window, args.alphabet_size)

  # Create the model
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer([args.alphabet_window * 2 + 1]))
  [model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation=activation)) for _ in range(args.hidden_layers)]
  model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

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
    data.train.data["windows"], data.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(data.dev.data["windows"], data.dev.data["labels"]),
    callbacks=[tb_callback],
  )

  # Evaluate the model
  test_logs = model.evaluate(
    data.dev.data["windows"], data.dev.data["labels"], batch_size=args.batch_size,
  )

  with open("predictions.txt", "w") as f:
    f.write("")

  with open("predictions.txt", "a") as f:
    predictions = model.predict(data.test.data["windows"])
    for i in range(predictions.shape[0]):
      f.write(data.test.text[i].lower() if predictions[i, 0] > predictions[i, 1] else data.test.text[i].upper())

  # Logs
  tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})
  with open("uppercase_model.out", "w") as out_file:
    print("{:.2f}".format(100 * test_logs[1]), file=out_file)
