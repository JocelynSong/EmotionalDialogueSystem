import os
import sys
import random
import argparse

import numpy as np
import tensorflow as tf

__author__ = "Jocelyn"

emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}
FLAGS = None


class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, input, W=None, b=None, activation=tf.nn.tanh):
        self.input = input

        if W is None:
            init_width = np.sqrt(6. / (n_in + n_out))
            W = tf.Variable(tf.random_uniform([n_in, n_out], -init_width, init_width), name="hidden_layer_W")
            if activation == tf.nn.sigmoid:
                W = tf.Variable(4 * tf.random_uniform([n_in, n_out], -init_width, init_width), name="hidden_layer_W")
        if b is None:
            b = tf.Variable(tf.zeros(n_out, ), name="hidden_layer_b")
        self.W = W
        self.b = b

        lin_out = tf.matmul(self.input, self.W) + self.b
        self.output = lin_out if activation is None else activation(lin_out)


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = tf.Variable(tf.zeros([n_in, n_out]), dtype=tf.float32, name="logistic_regression_w")
        self.b = tf.Variable(tf.zeros([n_out, ]), dtype=tf.float32, name="logistic_regression_b")
        self.p_y_given_x = tf.nn.softmax(tf.matmul(input, self.W) + self.b)

        self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        self.input = input

    def negative_log_likelihood(self, y):
        return -tf.reduce_mean(tf.log(self.p_y_given_x)[range(len(y)), y])

    def errors(self, y):
        return tf.reduce_mean(tf.not_equal(self.y_pred, y))


class MLP(object):
    def __init__(self, session, rng, n_in, n_hidden, n_out, batch_size, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001):
        self.batch_size = batch_size
        self.session = session
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_in])
        self.y = tf.placeholder(dtype=tf.int64, shape=[batch_size, ])

        self.hidden_layer = HiddenLayer(rng=rng,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        input=self.x,
                                        activation=tf.nn.tanh)

        self.softmax_w = tf.Variable(tf.zeros([n_hidden, n_out]), dtype=tf.float32, name="logistic_regression_w")
        self.softmax_b = tf.Variable(tf.zeros([n_out, ]), dtype=tf.float32, name="logistic_regression_b")

        self.p_y_given_x = tf.nn.softmax(tf.matmul(self.hidden_layer.output, self.softmax_w) + self.softmax_b)
        self.y_prediction = tf.argmax(self.p_y_given_x, axis=1)
        self.loss_likelihood = self.negative_log_likelihood(self.y)
        self.errors = self.errors(self.y)

        self.L1 = tf.reduce_sum(tf.abs(self.hidden_layer.W)) + tf.reduce_sum(tf.abs(self.softmax_w))
        self.L2_sqr = tf.reduce_sum(tf.square(self.hidden_layer.W)) + tf.reduce_sum(tf.square(self.softmax_w))
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.loss = self.loss_likelihood + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_factor = tf.constant(0.99, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def negative_log_likelihood(self, y):
        with tf.device("/cpu:0"):
            y_labels = tf.one_hot(indices=y, depth=6, on_value=1.0, off_value=0.0)
        likelihood = tf.reduce_sum(tf.multiply(tf.log(self.p_y_given_x), y_labels), axis=1)
        return -tf.reduce_mean(likelihood)

    def errors(self, y):
        return tf.reduce_mean(tf.cast(tf.not_equal(self.y_prediction, y), tf.float32))

    def step(self, train_x, train_y):
        input_feed = {self.x: train_x,
                      self.y: train_y}
        output_feed = [self.train, self.loss, self.loss_likelihood]

        result = self.session.run(output_feed, input_feed)
        loss, loss_likelihood = result[1], result[2]
        return loss, loss_likelihood

    def valid(self, train_x, train_y):
        input_feed = {self.x: train_x,
                      self.y: train_y}
        output_feed = [self.loss, self.loss_likelihood]

        result = self.session.run(output_feed, input_feed)
        loss, loss_likelihood = result[0], result[1]
        return loss, loss_likelihood

    def predict(self, test_x, test_y):
        input_feed = {self.x: test_x,
                      self.y: test_y}
        output_feed = [self.errors]

        result = self.session.run(output_feed, input_feed)
        pred_error = result[0]
        return pred_error

    def predict_labels(self, test_x):
        input_feed = {self.x: test_x}
        output_feed = [self.y_prediction]

        result = self.session.run(output_feed, input_feed)
        labels = result[0]
        return labels

    def get_batch(self, train_x, train_y, index):
        this_train_x = train_x[index * self.batch_size: (index + 1) * self.batch_size]
        this_train_y = train_y[index * self.batch_size: (index + 1) * self.batch_size]
        return this_train_x, this_train_y


def get_doc_embedding(words, word_dict, unk="</s>"):
    embedding = np.zeros([100])
    for word in words:
        if word in word_dict:
            word_embedding = word_dict[word]
        else:
            continue
            # word_embedding = word_dict[unk]
        # embedding += word_embedding * np.log(words_idf_scores[word])
        embedding += word_embedding
    return embedding


def get_data(lines, word_dict, unk="</s>"):
    data = []
    for line in lines:
        words = line.strip().split()
        embedding = get_doc_embedding(words, word_dict, unk)
        data.append(embedding)
    return data


def count_idf(dir_path, files):
    word_idf = {}
    total_number = len(files)
    print("There is total number of %d English files\n" % total_number)
    index = 0
    for file in files:
        index += 1
        if index % 1000 == 0:
            print("Now deal %d files\n" % index)
        filename = os.path.join(dir_path, file)
        f = open(filename, "r", encoding="utf-8")
        lines = f.readlines()[1:]
        words = []
        for line in lines:
            words.extend(line.strip().split())
        words = set(words)
        for word in words:
            if word in word_idf:
                word_idf[word] += 1
            else:
                word_idf[word] = 1
    for key, count in word_idf.items():
        word_idf[key] = count / float(total_number)
    return word_idf


def read_emotion_data_label(emotion_file, label_file):
    emotion_f = open(emotion_file, "r", encoding="utf-8")
    label_f = open(label_file, "r", encoding="utf-8")

    emotion_data = list()
    emotion_labels = list()

    for emotion_line, label_line in zip(emotion_f.readlines(), label_f.readlines()):
        data = emotion_line.strip()
        label = label_line.strip()

        if label in emotion_dict.keys():
            emotion_data.append(data)
            emotion_labels.append(emotion_dict[label])
    emotion_f.close()
    label_f.close()
    return emotion_data, emotion_labels


def get_train_valid_test_set(train_file_name, train_label_name, test_file_name, test_label_name):
    """

    :param train_file_name:
    :param train_label_name:
    :param test_file_name:
    :return:
    """
    lines, labels = read_emotion_data_label(train_file_name, train_label_name)

    split_length = int(len(lines) / 10)
    train_lines = lines[: split_length * 8]
    train_labels = labels[: split_length * 8]

    valid_lines = lines[split_length * 8: split_length * 9]
    valid_labels = labels[split_length * 8: split_length * 9]

    test_lines = lines[split_length * 9:]
    test_labels = labels[split_length * 9:]

    predict_lines, predict_labels = read_emotion_data_label(test_file_name, test_label_name)

    return train_lines, train_labels, valid_lines, valid_labels, test_lines, test_labels, predict_lines, predict_labels


def read_freq_words(filename):
    words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        lemmas = line.strip().split("|||")
        words.append(lemmas[0].strip())
    return words


def read_freq_word_embeddings(filename, freq_file, unk="<UNK>"):
    freq_words = read_freq_words(freq_file)
    freq_words.append(unk)

    embedding = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        word = words[0]
        if word not in freq_words:
            continue
        vector = [float(dim.strip()) for dim in words[1:]]
        embedding[word] = np.array(vector)
    return embedding


def read_word_embeddings(filename):
    embedding = {}
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        word = words[0]
        vector = [float(dim.strip()) for dim in words[1:]]
        embedding[word] = np.array(vector)
    return embedding


def align_test_batch(test_sens, batch_size):
    length = len(test_sens)
    if length % batch_size == 0:
        return test_sens
    else:
        remain = batch_size - length % batch_size
        for _ in range(remain):
            sample_id = random.choice(range(length))
            test_sens.append(test_sens[sample_id])
        return test_sens


def align_train_batch(train_sens, train_labels, batch_size):
    length = len(train_sens)
    if length % batch_size == 0:
        return train_sens, train_labels
    else:
        remain = batch_size - length % batch_size
        for _ in range(remain):
            sample_id = random.choice(range(length))
            train_sens.append(train_sens[sample_id])
            train_labels.append(train_labels[sample_id])
        return train_sens, train_labels


def save_label_file(test_label_file, test_labels):
    f = open(test_label_file, "w", encoding="utf-8")
    for label in test_labels:
        f.write(id2emotion[label])
        f.write("\n")
    f.close()


def compute_accuracy(true_test_labels, predict_labels):
    total_len = len(true_test_labels)
    predict_labels = predict_labels[:total_len]
    num = 0
    for true_label, pred_label in zip(true_test_labels, predict_labels):
        if true_label == pred_label:
            num += 1
    acc = float(num) / total_len
    print("Prediction accuracy: %f\n" % acc)


def train(train_file_name, train_label_name, test_file_name, test_label_file, word_embeddings, session):
    learning_rate = 0.0008
    l1_reg = 0.00
    l2_reg = 0.0001
    n_epochs = 1000
    batch_size = 32
    n_hidden = 512

    train_lines, train_labels, valid_lines, valid_labels, test_lines, test_labels, predict_lines, true_test_labels = \
        get_train_valid_test_set(train_file_name, train_label_name, test_file_name, test_label_file)

    train_x = get_data(train_lines, word_embeddings, FLAGS.unk)
    valid_x = get_data(valid_lines, word_embeddings, FLAGS.unk)
    test_x = get_data(test_lines, word_embeddings, FLAGS.unk)
    predict_x = get_data(predict_lines, word_embeddings, FLAGS.unk)

    train_x, train_labels = align_train_batch(train_x, train_labels, batch_size)
    valid_x, valid_labels = align_train_batch(valid_x, valid_labels, batch_size)
    test_x, test_labels = align_train_batch(test_x, test_labels, batch_size)
    predict_x = align_test_batch(predict_x, batch_size)

    n_train_batches = len(train_x) // batch_size
    n_valid_batches = len(valid_x) // batch_size
    n_test_batches = len(test_x) // batch_size

    print("... building the model\n")

    rng = np.random.RandomState(1234)

    classifier = MLP(session=session,
                     rng=rng,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=6,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     L1_reg=l1_reg,
                     L2_reg=l2_reg)

    print("Training!\n")

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_test_score = np.inf
    best_iter = 0
    test_score = 0.

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            this_train_x, this_train_y = classifier.get_batch(train_x, train_labels, minibatch_index)
            this_train_loss, this_train_likelihood = classifier.step(this_train_x, this_train_y)

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = []
                for valid_index in range(n_valid_batches):
                    this_valid_x, this_valid_y = classifier.get_batch(valid_x, valid_labels, valid_index)
                    this_valid_loss = classifier.predict(this_valid_x, this_valid_y)
                    validation_losses.append(this_valid_loss)
                this_ave_valid_loss = np.mean(validation_losses)

                print("epoch=%d, mini_batch=%d/%d, validation acc=%f %%\n" % (epoch, minibatch_index + 1,
                                                                              n_train_batches,
                                                                              (1.0 - this_ave_valid_loss) * 100))
                if this_ave_valid_loss < best_validation_loss:
                    if this_ave_valid_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        # session.run(classifier.learning_rate_decay_op)

                    best_validation_loss = this_ave_valid_loss
                    best_iter = iter

                    test_losses = []
                    for test_index in range(n_test_batches):
                        this_test_x, this_test_y = classifier.get_batch(test_x, test_labels, test_index)
                        this_valid_loss = classifier.predict(this_test_x, this_test_y)
                        test_losses.append(this_valid_loss)
                    # test_score = np.mean(test_losses)
                    test_score = np.min(test_losses)
                    if test_score < best_test_score:
                        best_test_score = test_score
                    print("epoch=%d, mini_batch=%d/%d, test acc of best model=%f %%" % (epoch, minibatch_index + 1,
                                                                                        n_train_batches,
                                                                                        (1.0 - test_score) * 100))
            if patience <= iter:
                done_looping = True
                break

    print("Optimization Complete. Best Validation score of %f %% obtained at iteration %d, with test performance %f %%"
          % ((1.0 - best_validation_loss) * 100 % 100, best_iter + 1, (1.0 - test_score) * 100.))

    n_predict_batches = len(predict_x) // batch_size
    test_labels = []
    for i in range(n_predict_batches):
        this_predict_x = predict_x[i * batch_size: (i + 1) * batch_size]
        predict_labels = classifier.predict_labels(this_predict_x)
        test_labels.extend(predict_labels)

    save_label_file(FLAGS.test_predict_label_file, test_labels)
    compute_accuracy(true_test_labels, test_labels)


def main(_):
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))

        print("read word embedding\n")
        word_embeddings = read_word_embeddings(FLAGS.embedding_path)

        print("training classifier\n")
        train(FLAGS.train_file_name, FLAGS.train_label_name, FLAGS.test_file_name, FLAGS.test_label_file,
              word_embeddings, sess)


if __name__ == "__main__":
    dir_path = "E:\\NLP\\code\\dialogue\\emotional chat machine\\data"
    test_dir = "E:\\NLP\\code\\dialogue\\experiment\\data"

    parse = argparse.ArgumentParser()

    parse.add_argument("--train_file_name", type=str,
                       default=os.path.join(dir_path, "clean_file\\emotion.data.seg.txt"),
                       help="source document file directory")
    parse.add_argument("--train_label_name", type=str,
                       default=os.path.join(dir_path, "clean_file\\emotion.label.txt"),
                       help="target document file directory")
    parse.add_argument("--test_file_name", type=str,
                       default=os.path.join(test_dir, "ECM\\new.test.res.txt"),
                       help="source embedding file path")
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(dir_path, "stc_data\\train_test\\test.label.txt"),
                       help="True test label file name")
    parse.add_argument("--test_predict_label_file", type=str,
                       default=os.path.join(test_dir, "visualization\\test.ecm.predict.label.txt"))
    parse.add_argument("--embedding_path", type=str,
                       default=os.path.join(dir_path, "embedding\\emotion.embeddings.txt"),
                       help="target embedding file path")

    parse.add_argument("--unk", type=str, default="</s>", help="unk label for this embedding")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
























