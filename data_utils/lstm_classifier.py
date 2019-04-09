import os
import sys
import random
import argparse

import tensorflow as tf
import numpy as np

from src.utils import matrix_initializer, truncated_normal_initializer_variable, zero_initializer_variable
from data_utils.prepare_dialogue_data import get_word_count, construct_word_dict

__author__ = "Jocelyn"

emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}
FLAGS = None


class LstmClassifier(object):
    def __init__(self, word_embeddings, words2idx, embedding_size, hidden_size, emotion_class, batch_size,
                 max_len, use_lstm, session, keep_prob=2.0, learning_rate=0.001, lr_decay=0.5, name="lstm_classifier"):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.emotion_class = emotion_class
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.sess = session
        self.keep_prob = keep_prob
        self.learning_rate_decay_factor = lr_decay
        self.name = name

        # word embeddings
        self.words2idx = words2idx
        self.idx2words = {idx: word for word, idx in self.words2idx.items()}
        self.embeddings = matrix_initializer(w=word_embeddings, name=self.name + "_word_embeddings")
        self.vocab_size = len(words2idx)

        # softmax
        self.sfx_w = truncated_normal_initializer_variable(width=hidden_size,
                                                           shape=[2 * self.hidden_size, self.emotion_class],
                                                           name=self.name+"_softmax_w")
        self.sfx_b = zero_initializer_variable(shape=[self.emotion_class], name=self.name+"_softmax_b")

        # placeholder
        self.input_x = tf.placeholder(shape=[self.batch_size, self.max_len], dtype=tf.int32, name=self.name+"_input_x")
        self.input_y = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_input_y")
        self.input_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_input_len")

        self.forward_cell = self.rnn_cell()
        self.backward_cell = self.rnn_cell()

        # loss
        self.loss = self.compute_loss()
        self.pred_scores = self.predict_scores()
        self.pred_labels = tf.argmax(self.pred_scores, axis=1)
        tf.summary.scalar("loss", self.loss)

        self.global_step = tf.Variable(0, name=self.name + "_global_step", trainable=False)
        self.lr = tf.Variable(learning_rate, dtype=tf.float32)
        self.train = self.optimize()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def basic_rnn_cell(self):
        if self.use_lstm:
            return tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.nn.rnn_cell.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)

    def rnn_cell(self):
        single_cell = self.basic_rnn_cell
        if self.keep_prob < 1.0:
            def single_cell():
                return tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
        cell = single_cell()
        # cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        return cell

    def lstm_process(self):
        input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_x)  # [batch, max_time, embedding size]

        initiate_state_forward = self.forward_cell.zero_state(self.batch_size, dtype=tf.float32)
        initiate_state_backward = self.backward_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.forward_cell, self.backward_cell, input_embeddings,
                                                          sequence_length=self.input_len,
                                                          initial_state_fw=initiate_state_forward,
                                                          initial_state_bw=initiate_state_backward,
                                                          dtype=tf.float32)
        output_fw_state, output_bw_state = states
        final_states = tf.concat([output_fw_state, output_bw_state], axis=-1)  # [2, batch, 2 * hidden]
        split_states_outputs = tf.split(final_states, num_or_size_splits=2, axis=0)
        final_states = tf.reshape(split_states_outputs[1], [self.batch_size, 2 * self.hidden_size])
        """
                outputs, states = tf.nn.dynamic_rnn(self.forward_cell, input_embeddings, sequence_length=self.input_len,
                                                    initial_state=initiate_state_forward, dtype=tf.float32)
                split_states_outputs = tf.split(states, num_or_size_splits=2, axis=0)
                final_states = tf.reshape(split_states_outputs[1], [self.batch_size, self.hidden_size])
                """
        return final_states

    def compute_loss(self):
        final_states = self.lstm_process()
        logits = tf.matmul(final_states, self.sfx_w) + self.sfx_b
        entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
        loss = tf.reduce_sum(entropy_loss) / self.batch_size
        return loss

    def predict_scores(self):
        final_states = self.lstm_process()
        logits = tf.matmul(final_states, self.sfx_w) + self.sfx_b
        scores = tf.nn.softmax(logits, dim=-1)
        return scores

    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        trainer = optimizer.minimize(self.loss)
        return trainer

    def train_step(self, this_input_x, this_input_y, this_input_len):
        output_feed = [self.train, self.loss]
        input_feed = {self.input_x: this_input_x,
                      self.input_y: this_input_y,
                      self.input_len: this_input_len}
        _, loss = self.sess.run(output_feed, input_feed)
        return loss

    def predict_step(self, this_input_x, this_input_len):
        output_feed = [self.pred_labels]
        input_feed = {self.input_x: this_input_x,
                      self.input_len: this_input_len}
        results = self.sess.run(output_feed, input_feed)
        return results[0]

    def get_train_batch(self, input_responses, input_labels, input_lengths, index):
        this_input_x = input_responses[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_y = input_labels[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_len = input_lengths[index * self.batch_size: (index + 1) * self.batch_size]
        return this_input_x, this_input_y, this_input_len

    def get_pred_batch(self, input_responses, input_lengths, index):
        this_input_x = input_responses[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_len = input_lengths[index * self.batch_size: (index + 1) * self.batch_size]
        return this_input_x, this_input_len


def read_emotional_response_label_file(train_res_file, train_label_file, max_len=30):
    f1 = open(train_res_file, "r", encoding="utf-8")
    f2 = open(train_label_file, "r", encoding="utf-8")
    res_lines = f1.readlines()
    label_lines = f2.readlines()

    train_responses = []
    train_labels = []
    train_lens = []
    for res_line, label_line in zip(res_lines, label_lines):
        label = label_line.strip()
        if label not in emotion_dict.keys():
            continue
        words = res_line.strip().split()
        if len(words) > max_len:
            words = words[: max_len]
        train_responses.append(words)
        train_labels.append(emotion_dict[label])
        train_lens.append(len(words))
    return train_responses, train_labels, train_lens


def read_test_data(filename, max_len):
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()

    responses = []
    train_lens = []
    for line in lines:
        words = line.strip().split()
        if len(words) > max_len:
            words = words[: max_len]
        responses.append(words)
        train_lens.append(len(words))
    return responses, train_lens


def response_to_indexs(train_responses, word_dict, word_unk_id, max_len):
    new_responses = []
    for response in train_responses:
        new_response = [word_dict[word] if word in word_dict else word_unk_id for word in response]
        if len(new_response) < max_len:
            remain = max_len - len(new_response)
            for i in range(remain):
                new_response.append(word_unk_id)
        new_responses.append(new_response)
    return new_responses


def align_train_batch_size(train_responses, emotion_labels, response_lens, batch_size):
    length = len(train_responses)
    if length % batch_size != 0:
        remain = batch_size - length % batch_size
        total_data = [[res, label, length] for res, label, length in
                      zip(train_responses, emotion_labels, response_lens)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        train_responses = [data[0] for data in total_data]
        emotion_labels = [data[1] for data in total_data]
        response_lens = [data[2] for data in total_data]
    return train_responses, emotion_labels, response_lens


def align_test_batch_size(train_responses, response_lens, batch_size):
    length = len(train_responses)
    if length % batch_size != 0:
        remain = batch_size - length % batch_size
        total_data = [[res, length] for res, length in
                      zip(train_responses, response_lens)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        train_responses = [data[0] for data in total_data]
        response_lens = [data[1] for data in total_data]
    return train_responses, response_lens


def shuffle_train_data(train_responses, emotion_labels, response_lens):
    total_data = [[res, label, length] for res, label, length in
                  zip(train_responses, emotion_labels, response_lens)]

    random.shuffle(total_data)
    train_responses = [data[0] for data in total_data]
    emotion_labels = [data[1] for data in total_data]
    response_lens = [data[2] for data in total_data]
    return train_responses, emotion_labels, response_lens


def write_labels(file_name, labels):
    f = open(file_name, "w", encoding="utf-8")
    for label in labels:
        f.write(id2emotion[label])
        f.write("\n")
    f.close()


def split_train_valid_data(train_responses, train_labels, train_lens):
    total_length = len(train_responses)
    train_len = int(total_length * 0.9)
    valid_len = total_length - train_len
    sequence = random.sample(range(total_length), valid_len)
    training_res, training_labels, training_lens, valid_res, valid_labels, valid_lens = [], [], [], [], [], []
    for i in range(total_length):
        if i in sequence:
            valid_res.append(train_responses[i])
            valid_labels.append(train_labels[i])
            valid_lens.append(train_lens[i])
        else:
            training_res.append(train_responses[i])
            training_labels.append(train_labels[i])
            training_lens.append(train_lens[i])
    return training_res, training_labels, training_lens, valid_res, valid_labels, valid_lens


def compute_accuracy(pred_labels, true_labels):
    total_len = len(pred_labels)
    num = 0
    for pred, true_label in zip(pred_labels, true_labels):
        if pred == true_label:
            num += 1
    acc = float(num) / total_len
    return acc


def read_total_embeddings(embedding_file, vocab_size):
    embeddings = list()
    word2id = dict()
    id2word = dict()
    f = open(embedding_file, "r", encoding="utf-8")
    for line in f.readlines()[: vocab_size]:
        lemmas = line.strip().split()
        word = lemmas[0].strip()
        embedding = list()
        for lemma in lemmas[1:]:
            embedding.append(float(lemma.strip()))
        index = len(word2id)
        word2id[word] = index
        id2word[index] = word
        embeddings.append(embedding)
    return embeddings, word2id, id2word


def train(train_response_file, train_label_file, test_res_file, test_label_file, max_len, word_count_file, vocab_size,
          embedding_file, embedding_size, batch_size, num_epoch, hidden_size, emotion_class, pred_label_file, session):
    print("read word and embeddings\n")
    embeddings, word_dict, id2word = read_total_embeddings(embedding_file, vocab_size)
    word_unk_id = word_dict[FLAGS.unk]

    print("reading training data\n")
    train_responses, train_labels, train_lens = read_emotional_response_label_file(train_response_file,
                                                                                   train_label_file,
                                                                                   max_len)
    train_responses = response_to_indexs(train_responses, word_dict, word_unk_id, max_len)
    train_responses, train_labels, train_lens, valid_res, valid_labels, valid_lens = \
        split_train_valid_data(train_responses, train_labels, train_lens)

    train_responses, train_labels, train_lens = align_train_batch_size(train_responses, train_labels, train_lens,
                                                                       batch_size)
    valid_res, valid_labels, valid_lens = align_train_batch_size(valid_res, valid_labels, valid_lens, batch_size)

    print("Read prediction data!\n")
    test_responses, test_labels, test_lens = read_emotional_response_label_file(test_res_file, test_label_file, max_len)
    test_responses = response_to_indexs(test_responses, word_dict, word_unk_id, max_len)
    test_length = len(test_responses)
    test_responses, test_labels, test_lens = align_train_batch_size(test_responses, test_labels, test_lens, batch_size)

    print("Define model!\n")
    lstm_emotion_machine = LstmClassifier(embeddings, word_dict, embedding_size, hidden_size, emotion_class, batch_size,
                                          max_len, True, session, learning_rate=FLAGS.learning_rate)

    print("training\n")
    train_batch = int(len(train_responses) / batch_size)
    valid_batch = int(len(valid_res) / batch_size)
    valid_accs = []
    best_valid_acc = -1.0
    ckpt_path = os.path.join(FLAGS.checkpoint_path, "lstm-classifier")
    for i in range(num_epoch):
        print("Now train epoch %d!\n" % (i + 1))
        train_responses, train_labels, train_lens = shuffle_train_data(train_responses, train_labels, train_lens)

        for j in range(train_batch):
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(train_responses, train_labels,
                                                                                  train_lens, j)
            loss = lstm_emotion_machine.train_step(this_res, this_label, this_len)
            print("epoch=%d, batch=%d, loss=%f\n" % ((i + 1), (j + 1), loss))

        labels = []
        for k in range(valid_batch):
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(valid_res, valid_labels,
                                                                                  valid_lens, k)
            this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
            labels.extend(this_labels)
        accuracy = compute_accuracy(labels, valid_labels)
        print("epoch=%d, accuracy=%f\n" % ((i + 1), accuracy))
        valid_accs.append(accuracy)

        if best_valid_acc < accuracy:
            best_valid_acc = accuracy
            lstm_emotion_machine.saver.save(lstm_emotion_machine.sess, ckpt_path, global_step=(i + 1) * train_batch)

    best_acc = np.max(valid_accs)
    ave_acc = np.average(valid_accs)
    print("best acc=%f, average acc=%f\n" % (best_acc, ave_acc))

    restore_path = lstm_emotion_machine.saver.last_checkpoints[-1]
    lstm_emotion_machine.saver.restore(lstm_emotion_machine.sess, restore_path)
    pred_batches = int(len(test_responses) / batch_size)
    total_labels = []
    for k in range(pred_batches):
        this_res, this_len = lstm_emotion_machine.get_pred_batch(test_responses, test_lens, k)
        this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
        total_labels.extend(this_labels)

    # write_labels(pred_label_file, total_labels)
    predict_labels = total_labels[: test_length]
    test_labels = test_labels[: test_length]
    test_acc = compute_accuracy(predict_labels, test_labels)
    print("test accuracy=%f\n" % test_acc)


def main(_):
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        train(FLAGS.train_response_file, FLAGS.train_label_file, FLAGS.test_response_file, FLAGS.test_label_file,
              FLAGS.max_len, FLAGS.word_count_file, FLAGS.vocab_size, FLAGS.embedding_file, FLAGS.embedding_size,
              FLAGS.batch_size, FLAGS.num_epoch, FLAGS.hidden_size, FLAGS.emotion_class, FLAGS.pred_label_file, sess)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("lstm_classifier.py")))
    data_dir = os.path.join(model_path, "data")

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_response_file", type=str,
                       default=os.path.join(data_dir, "clean_file\\emotion.data.seg.txt"))
    parse.add_argument("--train_label_file", type=str,
                       default=os.path.join(data_dir, "clean_file\\emotion.label.txt"))
    parse.add_argument("--test_response_file", type=str,
                       default=os.path.join(model_path, "exp\\test\\ablation\\ablation.response.1e-3.txt"))
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data\\train_test\\test.label.lstm.filter.txt"))
    parse.add_argument("--max_len", type=int, default=30)
    parse.add_argument("--word_count_file", type=str,
                       default=os.path.join(data_dir, "emotion_words_human\\word.count.txt"))
    parse.add_argument("--vocab_size", type=int, default=40000)
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(data_dir, "embedding\\emotion.embeddings.txt"),
                       help="word embedding file path")
    parse.add_argument("--embedding_size", type=int, default=100)
    parse.add_argument("--batch_size", type=int, default=64)
    parse.add_argument("--num_epoch", type=int, default=30)
    parse.add_argument("--hidden_size", type=int, default=256)
    parse.add_argument("--emotion_class", type=int, default=6)
    parse.add_argument("--pred_label_file", type=str, default="response.labels.txt")
    parse.add_argument("--unk", type=str, default="</s>", help="symbol for unk words")
    parse.add_argument("--learning_rate", type=float, default=0.1)
    parse.add_argument("--checkpoint_path", type=str, default=os.path.join(model_path, "data_utils\\check_path"))

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
















