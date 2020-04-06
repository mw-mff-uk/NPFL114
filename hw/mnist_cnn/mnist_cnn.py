#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        # TODO: Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # //- `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        # //  activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # //- `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output
        #   (after the ReLU nonlinearity of the last one).
        # //- `F`: Flatten inputs. Must appear exactly once in the architecture.
        # //- `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # //- `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in variable `hidden`.

        # print(re.split(",(?![^[]*\])", args.cnn))

        def res_net_block(input_data, inner_layers):
            residual = None
            for i, layer in enumerate(inner_layers):
                prev = input_data if residual is None else residual
                layer = layer.split("-")
                if (layer[0] == "C"):
                    residual = tf.keras.layers.Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], activation=tf.nn.relu)(prev)
                elif (layer[0] == "CB"):
                    residual = tf.keras.layers.Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], activation=None, use_bias=False)(prev)
                    residual = tf.keras.layers.BatchNormalization()(residual)
                    residual = tf.keras.layers.Activation(tf.nn.relu)(residual)
                elif (layer[0] == "M"):
                    residual = tf.keras.layers.MaxPooling2D(int(layer[1]), int(layer[2]))(prev)
                elif (layer[0] == "F"):
                    residual = tf.keras.layers.Flatten()(prev)
                    pass
                elif (layer[0] == "H"):
                    residual = tf.keras.layers.Dense(int(layer[1]), activation=tf.nn.relu)(prev)
                elif (layer[0] == "D"):
                    residual = tf.keras.layers.Dropout(float(layer[1]))(prev)
            residual = tf.keras.layers.Add()([residual, input_data])
            return residual


        hidden = inputs
        for i, layer in enumerate(list(re.split(",(?![^[]*\])", args.cnn))):
            if (layer[:2] != "R-"):
                layer = layer.split("-")
                
                if (layer[0] == "C"):
                    hidden = tf.keras.layers.Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], activation=tf.nn.relu)(hidden)
                elif (layer[0] == "CB"):
                    hidden = tf.keras.layers.Conv2D(int(layer[1]), int(layer[2]), int(layer[3]), layer[4], activation=None, use_bias=False)(hidden)
                    hidden = tf.keras.layers.BatchNormalization()(hidden)
                    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
                elif (layer[0] == "M"):
                    hidden = tf.keras.layers.MaxPooling2D(int(layer[1]), int(layer[2]))(hidden)
                elif (layer[0] == "F"):
                    hidden = tf.keras.layers.Flatten()(hidden)
                    pass
                elif (layer[0] == "H"):
                    hidden = tf.keras.layers.Dense(int(layer[1]), activation=tf.nn.relu)(hidden)
                elif (layer[0] == "D"):
                    hidden = tf.keras.layers.Dropout(float(layer[1]))(hidden)
            else:
                inner_layers = list(layer[3:-1].split(","))
                hidden = res_net_block(hidden, inner_layers)

                

                
            
        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self._tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def train(self, mnist, args):
        self._model.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self._tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self._model.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self._tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(self._model.metrics_names, test_logs)})
        return test_logs[self._model.metrics_names.index("accuracy")]


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
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
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
