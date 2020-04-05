#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Dropout, Input

from cifar10 import CIFAR10

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Training
  parser.add_argument("--batch_size", default=45, type=int, help="Batch size.")
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
  args = parser.parse_args([] if "__file__" not in globals() else None)

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
  os.mkdir(args.logdir)

  # Load data
  cifar = CIFAR10()

  inputs = Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
  hidden = inputs

  # Build the CNN
  with open("model.txt") as in_file:
    model_txt = in_file.read()
    for i, layer in enumerate(model_txt.split("\n")):
      layer = layer.split("-")

      print(f"Layer [{(i+1):02d}]: {'-'.join(layer)}")
      
      if (layer[0] == "C"):
        hidden = Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], kernel_initializer='he_uniform', activation=tf.nn.relu)(hidden)
      elif (layer[0] == "CB"):
        hidden = Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], kernel_initializer='he_uniform', activation=None, use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)
      elif (layer[0] == "M"):
        hidden = MaxPooling2D((int(layer[1]), int(layer[2])))(hidden)
      elif (layer[0] == "F"):
        hidden = Flatten()(hidden)
      elif (layer[0] == "H"):
        hidden = Dense(int(layer[1]), kernel_initializer='he_uniform', activation=tf.nn.relu)(hidden)
      elif (layer[0] == "D"):
        hidden = Dropout(float(layer[1]))(hidden)

    with open(os.path.join(args.logdir, "model.txt"), "w") as out_file:
      out_file.write(model_txt)
  
  outputs = Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  # Image data generator
  train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=args.rotation_range,
    zoom_range=args.zoom_range,
    width_shift_range=args.width_shift_range,
    height_shift_range=args.height_shift_range,
    horizontal_flip=bool(args.horizontal_flip)
  )

  # Tensor board
  tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

  # Fit the model
  model.fit(
    train_generator.flow(
      cifar.train.data["images"],
      cifar.train.data["labels"],
      batch_size=args.batch_size,
      seed=args.seed
    ),
    shuffle=False,
    epochs=args.epochs,
    validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
    callbacks=[tb_callback],
  )

  # Evaluate the model
  test_logs = model.evaluate(
    cifar.dev.data["images"], cifar.dev.data["labels"], batch_size=args.batch_size,
  )

  # Save the model
  model.save(os.path.join(args.logdir, "model.h5"))

  # Generate test set annotations, but in args.logdir to allow parallel execution.
  with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
    for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
      print(np.argmax(probs), file=out_file)
