from __future__ import division
from __future__ import print_function

import argparse
import sys

import NeuralNetwork as nn
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split


# a = tf.keras.datasets.mnist.load_data()

def main(batch_size, epochs, gd_variant, learning_rate, optimizer, layers, layer_size, activation, regularization):
    # shape = [784, 500, 100, 50, 30, 10]
    shape = [784] + list(range(layers * layer_size, layer_size - 1, -1 * layer_size)) + [10]
    regularization = regularization == "True"
    activations = {
        "sgm": nn.sgm,
        "relu": nn.relu,
        "tanh": nn.tanh
    }

    reg_tag = "reg" if regularization else "no_reg"

    name = "bs:" + str(batch_size) + "_epoch:" + str(epochs) + "_gd:" + str(gd_variant) + "_lr:" + str(learning_rate) + "_opt:" + str(
        optimizer) + "_num_lay:" + str(layers) + "_lay_size:" + str(layer_size) + "_act:" + str(
        activation) + "_" + reg_tag
    wandb.init(project="assignment1", name=name)

    net = nn.NeuralNetwork(shape, activation=activations[activation])

    train, test = tf.keras.datasets.fashion_mnist.load_data()
    X_train, y_train = train

    class_map = {}

    for index, img_class in enumerate(y_train):
        class_map[img_class] = index
        if len(class_map) == 10:
            break

    for i in range(10):
        plt.subplot(2, 5, 1 + i)
        plt.imshow(X_train[class_map[i]], cmap=plt.get_cmap('gray'))
    # plt.show()

    wandb.log({'class_sample_plot': plt})
    plt.clf()

    y_train = y_train.reshape(60000, 1)
    assert (X_train.shape, y_train.shape) == ((60000, 28, 28),
                                              (60000, 1)), "Train images were loaded incorrectly"
    X_train = X_train.reshape(60000, 784)

    X_test, y_test = test
    y_test = y_test.reshape(10000, 1)
    assert (X_test.shape, y_test.shape) == ((10000, 28, 28),
                                            (10000, 1)), "Test images were loaded incorrectly"
    X_test = X_test.reshape(10000, 784)

    (X_train, X_valid, y_train, y_valid) = train_test_split(X_train, y_train, stratify=y_train, test_size=10000)

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
        test_data=X_test,
        test_labels=y_test,
        valid_data=X_valid,
        valid_labels=y_valid,
        gd_variant=gd_variant,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        optimizer=optimizer
    )


if __name__ == '__main__':
    # main(sys.argv[1:])
    print(sys.argv[1:])
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Print or check SHA1 (160-bit) checksums."
    )
    parser.add_argument("--batch_size")
    parser.add_argument('--epochs', )
    parser.add_argument('--gd_variant', )
    parser.add_argument('--learning_rate', )
    parser.add_argument('--optimizer', )
    parser.add_argument('--layers', )
    parser.add_argument('--layer_size')
    parser.add_argument('--activation')
    parser.add_argument('--regularization')

    args = parser.parse_args()
    main(int(args.batch_size), int(args.epochs), args.gd_variant, float(args.learning_rate), args.optimizer,
         int(args.layers), int(args.layer_size), args.activation, args.regularization)
