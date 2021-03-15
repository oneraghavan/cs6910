from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, confusion_matrix
from tensorflow.keras import initializers

sns.set()
import plotly.express as px
import plotly.figure_factory as ff


def softmax(Z):
    expZ = np.exp(Z - np.max(Z.T, axis=1))
    return expZ / expZ.sum(axis=0, keepdims=True)


def sgm(x, der=False):
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)


def relu(x, der=False):
    x = np.divide(x, np.max(x.T, axis=1))

    @np.vectorize
    def relu1(x, der=False):
        if not der:
            return np.maximum(0, x)
        else:
            if x <= 0:
                return 0
            else:
                return 1

    return relu1(x, der)


# @np.vectorize
def tanh(z, der=False):
    # t = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    t = np.tanh(z)
    if der:
        return 1 - t ** 2
    return t


class NeuralNetwork:
    def __initialize_weights_and_biases(self, xavier):
        self.weights = [initializers.GlorotUniform()(shape=(j, i)).numpy() if xavier else np.random.randn(j, i) for i, j
                        in zip(
                self.shape[:-1], self.shape[1:])]
        self.biases = [np.random.randn(i, 1) for i in self.shape[1:]]

    def _initialize_activations_and_pre_activations(self, size=None):
        self.activations = [np.zeros((i, size))
                            for i in self.shape[1:]] if size else []
        self.pre_activations = [np.zeros((i, size))
                                for i in self.shape[1:]] if size else []

    def _init_deltas(self, size=None):
        self.deltas = [np.zeros((i, size))
                       for i in self.shape[1:]] if size else []

    def _init_dropout(self, size=None):
        self.dropout = [np.zeros((i, size))
                        for i in self.shape[1:]] if size else []

    def __init__(self, shape, activation=sgm, xavier_initialization=True):
        self.shape = shape
        self.activation = activation
        self.__initialize_weights_and_biases(xavier_initialization)
        self._initialize_activations_and_pre_activations()

    def vectorize_output(self):
        num_labels = np.unique(self.target).shape[0]
        num_examples = self.target.shape[1]
        result = np.zeros((num_labels, num_examples))
        for l, c in zip(self.target.ravel(), result.T):
            c[l] = 1
        self.target = result

    def labelize(self, data):
        return np.argmax(data, axis=0)

    def feed_forward(self, data, return_labels=False):

        self._initialize_activations_and_pre_activations()

        self.activations.append(data)
        self.pre_activations.append(data)
        result = data
        for item, value in enumerate(zip(self.weights, self.biases), start=1):
            w, b = value
            result = np.dot(w, result) + b

            self.pre_activations.append(result)
            if item == len(self.weights):
                result = softmax(result)
            else:
                result = self.activation(result)
            self.activations.append(result)

        if return_labels:
            result = self.labelize(result)

        # the last level is the activated output
        return result

    def compute_derivatives(self, data, target):
        self._init_deltas()

        self.deltas.append(self.activations[-1] - target)

        # since it's back propagation we start from the end
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                np.dot(
                    self.weights[l].T,
                    self.deltas[steps - l]
                ),
                self.activation(self.pre_activations[l], der=True)
            )
            self.deltas.append(delta)

        # delta[i] contains the delta for layer i+1
        self.deltas.reverse()

    def metrics(self, predicted, target):
        # the cost is normalized (divided by numer of samples)
        if predicted.shape != target.shape:
            target = target.reshape(predicted.shape)
        # return f1_score(target, predicted, average='micro')
        return precision_score(target, predicted, average='micro'), \
               recall_score(target, predicted, average='micro'), \
               f1_score(target, predicted, average='micro'), \
               accuracy_score(target, predicted)

    def cost(self, predicted, actuals):
        return log_loss(actuals, predicted.T)

    def train(self, train_data=None, train_labels=None, batch_size=100, epochs=20, learning_rate=.3, test_data=None,
              test_labels=None, valid_data=None, valid_labels=None, gd_variant='mini_batch', optimizer=None,
              regularization=True, regularization_lambda=0.5):

        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        wandb.config.batch_size = batch_size
        wandb.config.epochs = 20
        wandb.config.learning_rate = learning_rate
        wandb.config.gd_variant = gd_variant

        wandb.config.layers = len(self.shape) - 2
        wandb.config.layer_size = self.shape[-2]

        wandb.config.l2_regularization = regularization

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.batch_size = batch_size

        if not optimizer or optimizer.__class__ == VennilaOptimizer.__class__:
            self.optimizer = VennilaOptimizer(self)
        else:
            self.optimizer = optimizer

        wandb.config.optimizer = optimizer.__class__
        gamma = None
        if 'gamma' in dir(optimizer):
            gamma = optimizer.gamma
        wandb.config.optimizer_gamma = gamma

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        self.data = train_data.T
        self.target = train_labels.T
        self.original_labels = self.target.ravel()
        self.vectorize_output()

        self.validate_data()

        self.test_data = test_data.T
        self.test_labels = test_labels.reshape((10000,))

        self.valid_data = valid_data.T
        self.valid_labels = valid_labels.reshape((10000,))

        self.number_of_examples = self.data.shape[1]
        diff = self.number_of_examples % batch_size
        # we discard the last examples for now
        data = self.data
        target = self.target
        if diff != 0:
            data = self.data[: self.number_of_examples - diff]
            target = self.target[: self.number_of_examples - diff]
            self.number_of_examples = self.data.shape[1]

        for epoch in range(epochs):

            print("epoch:", epoch + 1, "/", epochs, end=" ")

            if gd_variant == 'stochastic':
                batches_input, batches_target = self.get_batches(1, data, target)
                batch_size = 1
            elif gd_variant == 'mini_batch':
                batches_input, batches_target = self.get_batches(batch_size, data, target)
            else:
                batches_input, batches_target = self.get_batches(len(train_data), data, target)
                batch_size = len(train_data)

            for batch_input, batch_target in zip(
                    batches_input, batches_target):
                self._initialize_activations_and_pre_activations()

                self.optimizer.gradient_decent(batch_input, batch_target)

            precision, recall, f1, accuracy = self.metrics(
                self.feed_forward(self.data, return_labels=True),
                self.original_labels
            )

            loss = self.cost(
                self.feed_forward(self.data),
                self.original_labels
            )

            wandb.log({'epoch': epoch, 'train_loss': loss, 'train_precision': precision, 'train_recall': recall,
                       'train_f1': f1, 'train_accuracy': accuracy})

            print(f"Train : loss {loss} precision {precision} , recall {recall} F1 Score {f1} accuracy {accuracy}")

            precision, recall, f1, accuracy = self.metrics(
                self.feed_forward(
                    self.valid_data, return_labels=True),
                self.valid_labels
            )

            loss = self.cost(
                self.feed_forward(self.valid_data),
                self.valid_labels
            )
            wandb.log({'epoch': epoch, 'valid_loss': loss, 'valid_precision': precision, 'valid_recall': recall,
                       'valid_f1': f1, 'valid_accuracy': accuracy})

            print(f"Test : loss {loss} precision {precision} , recall {recall} F1 Score {f1} accuracy {accuracy}")
            print()

        predictions = self.feed_forward(
            self.test_data, return_labels=True)

        frame = pd.DataFrame(confusion_matrix(test_labels, predictions))

        precision, recall, f1, accuracy = self.metrics(
            self.feed_forward(
                self.test_data, return_labels=True),
            self.test_labels
        )

        loss = self.cost(
            self.feed_forward(self.test_data),
            self.test_labels
        )
        wandb.log({'test_loss': loss, 'test_precision': precision, 'test_recall': recall,
                   'test_f1': f1, 'test_accuracy': accuracy})

        self.plot_confusion_matrixes(frame)

    def l2_regularization(self, w):
        if self.regularization:
            return self.learning_rate * self.regularization_lambda * w / self.batch_size
        else:
            return 0

    def plot_confusion_matrixes(self, plotdata):
        labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                  "Ankle boot"]
        confusions = plotdata.values
        plotdata.index = labels

        plotdata.columns = labels
        plotdata['Actual'] = labels

        fig = px.bar(plotdata, x="Actual", y=labels, title="Confusion matrix", labels={'value': "Predicted"})
        wandb.log({'confusion_stacked_bar': fig})
        del plotdata["Actual"]

        np.fill_diagonal(confusions, 0)
        plotdata = pd.DataFrame(confusions)
        plotdata.index = labels

        plotdata.columns = labels
        plotdata['Actual'] = labels

        fig = px.bar(plotdata, x="Actual", y=labels, title="Confusion matrix", labels={'value': "Predicted"})
        wandb.log({'mistakes_stacked_bar': fig})

        del plotdata["Actual"]
        fig = ff.create_annotated_heatmap(plotdata.values, x=labels, y=labels)
        wandb.log({'standard_confusion_matrix': fig})

    def validate_data(self):
        assert self.data.shape[0] == self.shape[0], \
            ('Input and shape of the network not compatible: ', self.data.shape[0], " != ", self.shape[0])
        assert self.target.shape[0] == self.shape[-1], \
            ('Output and shape of the network not compatible: ', self.target.shape[0], " != ", self.shape[-1])

    def get_batches(self, batch_size, data, target):
        batches_input = [data[:, k:k + batch_size]
                         for k in range(0, self.number_of_examples, batch_size)]
        batches_target = [target[:, k:k + batch_size]
                          for k in range(0, self.number_of_examples, batch_size)]
        return batches_input, batches_target


class VennilaOptimizer:

    def __init__(self, model):
        self.model = model

    def update_weights(self):
        self.model.weights = [
            w - ((self.model.learning_rate / self.model.batch_size) * np.dot(d, a.T)) - self.model.l2_regularization(w)
            for w, d, a in zip(self.model.weights, self.model.deltas, self.model.activations)]
        # print(self.model.weights[0][0],self.model.deltas[0])
        # self.model.weights = [w - (self.model.learning_rate / self.model.batch_size) * np.dot(d, a.T) * (2 * 0.5) * w
        #                       for w, d, a in zip(self.model.weights, self.model.deltas, self.model.activations)]

    def update_biases(self):
        self.model.biases = [
            b - (self.model.learning_rate / self.model.batch_size) * (np.sum(d, axis=1)).reshape(b.shape)
            for b, d in zip(self.model.biases, self.model.deltas)]

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()


class Momentum(VennilaOptimizer):

    def __init__(self, model, gamma=0.9):
        super(Momentum, self).__init__(model)
        self.gamma = gamma
        self.prev_weight_updates = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.prev_bias_updates = [np.zeros((i, 1)) for i in self.model.shape[1:]]

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()

    def update_weights(self):
        current_update = [(self.gamma * pd) + (self.model.learning_rate / self.model.batch_size) * np.dot(d, a.T)
                          for d, pd, a in
                          zip(self.model.deltas, self.prev_weight_updates, self.model.activations)]
        self.prev_weight_updates = current_update
        self.model.weights = [w - update - self.model.l2_regularization(w) for w, update in
                              zip(self.model.weights, current_update)]

    def update_biases(self):
        current_update = [
            (self.gamma * pb) + (self.model.learning_rate / self.model.batch_size) * (np.sum(d, axis=1)).reshape(
                b.shape)
            for b, pb, d in zip(self.model.biases, self.prev_bias_updates, self.model.deltas)]

        self.prev_bias_updates = current_update

        self.model.biases = [
            b - update
            for b, update in zip(self.model.biases, current_update)]


class NesterovAccelarated(VennilaOptimizer):

    def __init__(self, model, gamma):
        super(NesterovAccelarated, self).__init__(model)
        self.gamma = gamma
        self.prev_weight_updates = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.prev_bias_updates = [np.zeros((i, 1)) for i in self.model.shape[1:]]

    def gradient_decent(self, batch_input, batch_target):
        self.old_weights = self.model.weights
        self.old_biases = self.model.biases
        self.model.weights = [w - self.gamma * update - self.model.l2_regularization(w) for w, update in
                              zip(self.model.weights, self.prev_weight_updates)]
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()

    def update_weights(self):
        self.prev_weight_updates = [
            (self.gamma * pd) + (self.model.learning_rate / self.model.batch_size) * np.dot(d, a.T)
            for d, pd, a in
            zip(self.model.deltas, self.prev_weight_updates, self.model.activations)]

        self.model.weights = [
            w - (self.model.learning_rate / self.model.batch_size) * np.dot(d, a.T) - self.model.l2_regularization(w)
            for w, d, a in zip(self.old_weights, self.model.deltas, self.model.activations)]

    def update_biases(self):
        self.prev_bias_updates = [
            (self.gamma * pb) + (self.model.learning_rate / self.model.batch_size) * (np.sum(d, axis=1)).reshape(
                b.shape)
            for b, pb, d in zip(self.model.biases, self.prev_bias_updates, self.model.deltas)]

        self.model.biases = [
            b - (self.model.learning_rate / self.model.batch_size) * (np.sum(d, axis=1)).reshape(b.shape)
            for b, d in zip(self.old_biases, self.model.deltas)]


class Adagrad(VennilaOptimizer):

    def __init__(self, model, eps=1e-8):
        super(Adagrad, self).__init__(model)
        self.vw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.vb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.eps = eps

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()

    def update_weights(self):
        self.vw = [v + np.power(np.dot(d, a.T), 2) for v, d, a in
                   zip(self.vw, self.model.deltas, self.model.activations)]
        self.model.weights = [
            w - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vw + self.eps))) * np.dot(d,
                                                                                                       a.T) - self.model.l2_regularization(
                w)
            for w, d, a, vw in zip(self.model.weights, self.model.deltas, self.model.activations, self.vw)]

    #
    def update_biases(self):
        self.vb = [v + np.power(np.sum(d, axis=1).reshape(v.shape), 2) for v, d in zip(self.vb, self.model.deltas)]

        self.model.biases = [
            b - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vb + self.eps))) * (
                np.sum(d, axis=1)).reshape(b.shape)
            for b, d, vb in zip(self.model.biases, self.model.deltas, self.vb)]


class RMSProp(VennilaOptimizer):

    def __init__(self, model, beta=0.9, eps=1e-8):
        super(RMSProp, self).__init__(model)
        self.vw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.vb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.eps = eps
        self.beta = beta

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()

    def update_weights(self):
        self.vw = [(self.beta * v) + ((1 - self.beta) * np.power(np.dot(d, a.T), 2)) for v, d, a in
                   zip(self.vw, self.model.deltas, self.model.activations)]
        self.model.weights = [
            w - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vw + self.eps))) * np.dot(d,
                                                                                                       a.T) - self.model.l2_regularization(
                w)
            for w, d, a, vw in zip(self.model.weights, self.model.deltas, self.model.activations, self.vw)]

    #
    def update_biases(self):
        self.vb = [(self.beta * v) + ((1 - self.beta) * np.power(np.sum(d, axis=1).reshape(v.shape), 2)) for v, d in
                   zip(self.vb, self.model.deltas)]

        self.model.biases = [
            b - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vb + self.eps))) * (
                np.sum(d, axis=1)).reshape(b.shape)
            for b, d, vb in zip(self.model.biases, self.model.deltas, self.vb)]


class Adam(VennilaOptimizer):

    def __init__(self, model, beta1=0.9, beta2=.999, eps=1e-8):
        super(Adam, self).__init__(model)
        self.mw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.vw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.mb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.vb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.mw_hat = None
        self.vw_hat = None
        self.mb_hat = None
        self.vb_hat = None
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.acc_beta1 = beta1
        self.acc_beta2 = beta2

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()
        self.acc_beta1 = self.acc_beta1 * self.beta1
        self.acc_beta2 = self.acc_beta2 * self.beta2

    def update_weights(self):
        self.mw = [(self.beta1 * mw) + ((1 - self.beta1) * np.dot(d, a.T)) for mw, d, a in
                   zip(self.mw, self.model.deltas, self.model.activations)]

        self.vw = [(self.beta2 * vw) + ((1 - self.beta2) * np.power(np.dot(d, a.T), 2)) for vw, d, a in
                   zip(self.vw, self.model.deltas, self.model.activations)]

        self.mw_hat = [np.divide(mw, (1 - self.acc_beta1)) for mw in self.mw]

        self.vw_hat = [np.divide(vw, (1 - self.acc_beta1)) for vw in self.vw]

        self.model.weights = [
            w - (self.model.learning_rate / (
                        self.model.batch_size * np.sqrt(vw + self.eps))) * mw - self.model.l2_regularization(w)
            for w, vw, mw in zip(self.model.weights, self.vw_hat, self.mw_hat)]

    #
    def update_biases(self):
        self.mb = [(self.beta1 * mb) + ((1 - self.beta1) * np.sum(d, axis=1).reshape(mb.shape)) for mb, d in
                   zip(self.mb, self.model.deltas)]

        self.vb = [(self.beta2 * vb) + ((1 - self.beta2) * np.power(np.sum(d, axis=1).reshape(vb.shape), 2)) for vb, d
                   in
                   zip(self.vb, self.model.deltas)]
        # print()

        self.mb_hat = [np.divide(mb, (1 - self.acc_beta1)) for mb in self.mb]

        self.vb_hat = [np.divide(vb, (1 - self.acc_beta1)) for vb in self.vb]

        self.model.biases = [
            b - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vb + self.eps))) * mb
            for b, vb, mb in zip(self.model.biases, self.vb_hat, self.mb_hat)]


class NAdam(VennilaOptimizer):
    # https://ruder.io/optimizing-gradient-descent/index.html#nadam
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super(NAdam, self).__init__(model)
        self.mw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.vw = [np.zeros((j, i)) for i, j in zip(
            self.model.shape[:-1], self.model.shape[1:])]
        self.mb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.vb = [np.zeros((i, 1)) for i in self.model.shape[1:]]
        self.mw_hat = None
        self.vw_hat = None
        self.mb_hat = None
        self.vb_hat = None
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.acc_beta1 = beta1
        self.acc_beta2 = beta2

    def gradient_decent(self, batch_input, batch_target):
        self.model.feed_forward(batch_input)

        self.model.compute_derivatives(batch_input, batch_target)

        self.update_weights()
        self.update_biases()
        self.acc_beta1 = self.acc_beta1 * self.beta1
        self.acc_beta2 = self.acc_beta2 * self.beta2

    def update_weights(self):
        self.mw = [(self.beta1 * mw) + ((1 - self.beta1) * np.dot(d, a.T)) for mw, d, a in
                   zip(self.mw, self.model.deltas, self.model.activations)]

        self.vw = [(self.beta2 * vw) + ((1 - self.beta2) * np.power(np.dot(d, a.T), 2)) for vw, d, a in
                   zip(self.vw, self.model.deltas, self.model.activations)]

        self.mw_hat = [np.divide(mw, (1 - self.acc_beta1)) for mw in self.mw]

        self.vw_hat = [np.divide(vw, (1 - self.acc_beta1)) for vw in self.vw]

        self.model.weights = [
            w - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vw + self.eps))) * (
                    self.beta1 * mw + (1 - self.beta1) * np.dot(d, a.T) / (
                        1 - self.acc_beta1)) - self.model.l2_regularization(w)
            for w, vw, mw, d, a in
            zip(self.model.weights, self.vw_hat, self.mw_hat, self.model.deltas, self.model.activations)]

    #
    def update_biases(self):
        self.mb = [(self.beta1 * mb) + ((1 - self.beta1) * np.sum(d, axis=1).reshape(mb.shape)) for mb, d in
                   zip(self.mb, self.model.deltas)]

        self.vb = [(self.beta2 * vb) + ((1 - self.beta2) * np.power(np.sum(d, axis=1).reshape(vb.shape), 2)) for vb, d
                   in
                   zip(self.vb, self.model.deltas)]
        # print()

        self.mb_hat = [np.divide(mb, (1 - self.acc_beta1)) for mb in self.mb]

        self.vb_hat = [np.divide(vb, (1 - self.acc_beta1)) for vb in self.vb]

        self.model.biases = [
            b - (self.model.learning_rate / (self.model.batch_size * np.sqrt(vb + self.eps))) * (
                    self.beta1 * mb + (1 - self.beta1) * np.sum(d, axis=1).reshape(vb.shape) / (1 - self.acc_beta1))
            for b, vb, mb, d in zip(self.model.biases, self.vb_hat, self.mb_hat, self.model.deltas)]
