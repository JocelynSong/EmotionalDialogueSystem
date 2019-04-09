import tensorflow as tf

__author__ = "Jocelyn"


class Activation(object):
    def __init__(self, method):
        self.method = method.lower()

        if self.method == "sigmoid":
            self.func = tf.nn.sigmoid
        elif self.method == "tanh":
            self.func = tf.nn.tanh
        elif self.method == "relu":
            self.func = tf.nn.relu
        elif self.method == "elu":
            self.func = tf.nn.elu
        elif self.method == "identity":
            self.func = tf.identity
        else:
            raise ValueError("No such method name: %s\n" % self.method)

    def activate(self, x):
        return self.func(x)


