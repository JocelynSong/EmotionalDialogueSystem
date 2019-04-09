import configparser

import tensorflow as tf

from src.activation import Activation

__author__ = "Jocelyn"


class ChatConfig(object):
    def __init__(self, config_file):
        self.cf_parser = configparser.ConfigParser()
        self.cf_parser.read(config_file)

        (self.activation_name, self.embedding_size, self.hidden_size, self.emotion_vocab_size, self.normalize,
         self.weight_embedding, self.min_len, self.max_len, self.use_lstm, self.keep_prob, self.num_layers,
         self.emotion_class, self.interactive, self.statistics_interval, self.summary_interval,
         self.checkpoint_interval, self.softmax_smooth, self.word_count, self.generic_word_size, self.beam_size,
         self.lambda_reg, self.epochs_to_train, self.batch_size, self.concurrent_steps, self.uniform_width,
         self.random_seed, self.learning_rate_decay_factor) = self.parse()

        self.activation = Activation(self.activation_name)
        self.optimizer = OptimizerConfig(config_file)

    def parse(self):
        activation = self.cf_parser.get("function", "activation")

        embedding_size = self.cf_parser.getint("architectures", "embedding_size")
        hidden_size = self.cf_parser.getint("architectures", "hidden_size")
        emotion_vocab_size = self.cf_parser.getint("architectures", "emotion_vocab_size")
        normalize = self.cf_parser.getboolean("architectures", "normalize")
        weight_embedding = self.cf_parser.getfloat("architectures", "weight_embedding")
        min_len = self.cf_parser.getint("architectures", "min_len")
        max_len = self.cf_parser.getint("architectures", "max_len")
        use_lstm = self.cf_parser.getboolean("architectures", "use_lstm")
        keep_prob = self.cf_parser.getfloat("architectures", "keep_prob")
        num_layers = self.cf_parser.getint("architectures", "num_layers")
        emotion_class = self.cf_parser.getint("architectures", "emotion_class")
        interactive = self.cf_parser.getboolean("architectures", "interactive")
        statistics_interval = self.cf_parser.getint("architectures", "statistics_interval")
        summary_interval = self.cf_parser.getint("architectures", "summary_interval")
        checkpoint_interval = self.cf_parser.getint("architectures", "checkpoint_interval")
        softmax_smooth = self.cf_parser.getfloat("architectures", "softmax_smooth")
        word_count = self.cf_parser.getint("architectures", "word_count")
        generic_word_size = self.cf_parser.getint("architectures", "generic_word_size")
        beam_size = self.cf_parser.getint("architectures", "beam_size")
        lambda_reg = self.cf_parser.getfloat("architectures", "lambda_reg")

        epochs_to_train = self.cf_parser.getint("parameters", "epochs_to_train")
        batch_size = self.cf_parser.getint("parameters", "batch_size")
        concurrent_steps = self.cf_parser.getint("parameters", "concurrent_steps")
        uniform_width = self.cf_parser.getfloat("parameters", "uniform_width")
        random_seed = self.cf_parser.getfloat("parameters", "random_seed")

        learning_rate_decay_factor = self.cf_parser.getfloat("optimizer", "learning_rate_decay_factor")

        return (activation, embedding_size, hidden_size, emotion_vocab_size, normalize, weight_embedding, min_len,
                max_len, use_lstm, keep_prob, num_layers, emotion_class, interactive, statistics_interval,
                summary_interval, checkpoint_interval, softmax_smooth, word_count, generic_word_size, beam_size,
                lambda_reg, epochs_to_train, batch_size, concurrent_steps, uniform_width, random_seed,
                learning_rate_decay_factor)


class OptimizerConfig(object):
    def __init__(self, filename):
        self.cf_parser = configparser.ConfigParser()
        self.cf_parser.read(filename)
        self.opt_name, self.param = self.parse()
        self.lr = self.param["lr"]

    def parse(self):
        optimizer = self.cf_parser.get("optimizer", "optimizer")
        opt_param = self.get_opt_param(optimizer)
        return optimizer, opt_param

    def get_opt_param(self, optimizer):
        opt_param_dict = dict()
        opt_name = optimizer.lower()

        if opt_name == "sgd":
            opt_param_dict["lr"] = self.cf_parser.getfloat("optimizer", "lr")
        elif opt_name == "sgdmomentum":
            opt_param_dict["lr"] = self.cf_parser.getfloat("optimizer", "lr")
            opt_param_dict["momentum"] = self.cf_parser.getfloat("optimizer", "momentum")
        elif opt_name == "adagrad":
            opt_param_dict["lr"] = self.cf_parser.getfloat("optimizer", "lr")
        elif opt_name == "adadelta":
            opt_param_dict["lr"] = self.cf_parser.getfloat("optimizer", "lr")
            opt_param_dict["decay_rate"] = self.cf_parser.getfloat("optimizer", "learning_rate_decay_factor")
        elif opt_name == "adam":
            opt_param_dict["lr"] = self.cf_parser.getfloat("optimizer", "lr")
        else:
            raise ValueError("No such optimization name:%s\n" % opt_name)
        return opt_param_dict

    def get_optimizer(self):
        opt_name = self.opt_name.lower()

        if opt_name == "sgd":
            return tf.train.GradientDescentOptimizer(self.lr)
        elif opt_name == "sgdmomentum":
            return tf.train.MomentumOptimizer(self.lr, self.param["momentum"])
        elif opt_name == "adagrad":
            return tf.train.AdagradOptimizer(self.lr)
        elif opt_name == "adadelta":
            return tf.train.AdadeltaOptimizer(self.lr, self.param["decay_rate"])
        elif opt_name == "adam":
            return tf.train.AdamOptimizer(self.lr)
        else:
            raise ValueError("No such optimizer name: %s\n" % opt_name)





