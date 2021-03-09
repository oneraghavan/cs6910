from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import NeuralNetwork as nn


# a = tf.keras.datasets.mnist.load_data()

def main(batch_size, epochs, gd_variant, learning_rate, optimizer, layers, layer_size, activation):
    # shape = [784, 500, 100, 50, 30, 10]
    shape = [784] + list(range(layers * layer_size, layer_size - 1, -1 * layer_size)) + [10]

    activations = {
        "sgm": nn.sgm,
        "relu": nn.relu
    }

    net = nn.NeuralNetwork(shape, activation=activations[activation])

    train, test = tf.keras.datasets.fashion_mnist.load_data()

    X_train, y_train = train
    y_train = y_train.reshape(60000, 1)
    assert (X_train.shape, y_train.shape) == ((60000, 28, 28),
                                              (60000, 1)), "Train images were loaded incorrectly"
    X_train = X_train.reshape(60000, 784)

    X_test, y_test = test
    y_test = y_test.reshape(10000, 1)
    assert (X_test.shape, y_test.shape) == ((10000, 28, 28),
                                            (10000, 1)), "Test images were loaded incorrectly"
    X_test = X_test.reshape(10000, 784)

    optimizers = {
        "Adam": nn.Adam(net),
        "NAdam": nn.NAdam(net),
        "Momentum": nn.Momentum(net),
        "RMSProp": nn.RMSProp(net),
        "Adagrad": nn.Adagrad(net),
    }

    optimizer = optimizers[optimizer] if optimizer in optimizers.keys() else None

    net.train(
        train_data=X_train,
        train_labels=y_train,
        gd_variant=gd_variant,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        test_data=X_test,
        test_labels=y_test,
        optimizer=optimizer
    )


if __name__ == '__main__':
    # main(sys.argv[1:])
    print(sys.argv[1:])
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Print or check SHA1 (160-bit) checksums."
    )
    parser.add_argument(
        "--batch_size"
    )
    parser.add_argument('--epochs', )
    parser.add_argument('--gd_variant', )
    parser.add_argument('--learning_rate', )
    parser.add_argument('--optimizer', )
    parser.add_argument('--layers', )
    parser.add_argument('--layer_size')
    parser.add_argument('--activation')

    args = parser.parse_args()
    main(int(args.batch_size), int(args.epochs), args.gd_variant, float(args.learning_rate), args.optimizer,
         int(args.layers), int(args.layer_size), args.activation)
