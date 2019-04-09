import random
import math

import tensorflow as tf
import numpy as np
import logging

__author__ = "Jocelyn"


def uniform_initializer_variable(width, shape, name=""):
    var = tf.Variable(tf.random_uniform(shape, -width, width), name=name)
    return var


def truncated_normal_initializer_variable(width, shape, name=""):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=1.0 / tf.sqrt(float(width))), name=name, dtype=tf.float32)
    return var


def zero_initializer_variable(shape, name=""):
    var = tf.Variable(tf.zeros(shape=shape), name=name)
    return var


def matrix_initializer(w, name=""):
    var = tf.Variable(w, name=name)
    return var


def pre_logger(log_file_name, file_log_level=logging.DEBUG, screen_log_level=logging.INFO):
    """
    set log format
    :param log_file_name:
    :param file_log_level:
    :param screen_log_level:
    :return:
    """
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger(log_file_name)
    init_logger.setLevel(logging.DEBUG)

    # file handler
    file_handler = logging.FileHandler("log/{}.log".format(log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_log_level)

    # screen handler
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_log_level)

    init_logger.addHandler(file_handler)
    init_logger.addHandler(screen_handler)
    return init_logger



