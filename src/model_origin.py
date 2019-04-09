import tensorflow as tf
import numpy as np

from src.configuration import ChatConfig
from src.utils import matrix_initializer, truncated_normal_initializer_variable, zero_initializer_variable
from src.utils import uniform_initializer_variable

__author__ = "Song"

# "anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5


class EmotionChatMachine(object):
    def __init__(self, config_file, session, words2idx, word_embeddings, generic_word_length, start_id, end_id,
                 name=""):
        self.config = ChatConfig(config_file)
        self.embedding_size = self.config.embedding_size
        self.hidden_size = self.config.hidden_size
        self.emotion_vocab_size = self.config.emotion_vocab_size
        self.emotion_class = self.config.emotion_class
        self.batch_size = self.config.batch_size
        self.beam_size = self.config.beam_size
        self.lambda_reg = self.config.lambda_reg
        self.session = session
        self.start_id = start_id
        self.end_id = end_id
        self.name = name

        # generic embeddings & emotion embeddings
        self.words2idx = words2idx
        self.idx2words = {idx: word for word, idx in self.words2idx.items()}
        self.embeddings = matrix_initializer(w=word_embeddings, name=self.name+"_word_embeddings")
        self.num_generic_word = generic_word_length
        self.num_total_word = len(words2idx)  # total = generic + emotion class * emotion words

        # post attention weight
        self.post_trans_w = uniform_initializer_variable(width=3.0/self.hidden_size, shape=[self.hidden_size, 1],
                                                         name=self.name+"_post_transform_w")
        self.response_trans_w = uniform_initializer_variable(width=3.0/self.hidden_size,
                                                             shape=[self.hidden_size, 1],
                                                             name=self.name+"_response_transform_w")

        # emotion attention weight
        self.emotion_post_trans_w = uniform_initializer_variable(width=3.0/self.hidden_size,
                                                                 shape=[self.hidden_size, 1],
                                                                 name=self.name+"_emotion_post_transform_w")
        self.emotion_response_trans_w = uniform_initializer_variable(width=3.0/self.hidden_size,
                                                                     shape=[self.hidden_size, 1],
                                                                     name=self.name+"_emotion_res_transform_w")
        self.emotion_word_trans_w = uniform_initializer_variable(width=3.0/self.embedding_size,
                                                                 shape=[self.embedding_size, 1],
                                                                 name=self.name+"_emotion_word_transform_w")

        # softmax
        self.sfx_score_w = uniform_initializer_variable(width=3.0/self.hidden_size, shape=[self.hidden_size, 1],
                                                        name=self.name + "_softmax_score_weight")
        self.generic_word_mask = tf.constant(self.get_generic_word_mask(), dtype=tf.float32)
        self.sfx_w = uniform_initializer_variable(width=3.0/self.hidden_size,
                                                  shape=[self.hidden_size, self.num_total_word],
                                                  name=self.name+"_softmax_weight")
        self.sfx_b = zero_initializer_variable(shape=[1, self.num_total_word], name=self.name+"_softmax_bias")

        # regularization emotion
        self.reg_emotion_w = uniform_initializer_variable(width=3.0/self.embedding_size,
                                                          shape=[self.embedding_size, self.emotion_class + 1],
                                                          name=self.name+"_reg_emotion_w")
        self.reg_emotion_b = zero_initializer_variable(shape=[self.emotion_class + 1], name=self.name+"_reg_emotion_b")

        # input placeholder
        self.input_x = tf.placeholder(shape=[self.batch_size, self.config.max_len], dtype=tf.int32,
                                      name=self.name+"_input_x")
        self.input_length = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_input_length")
        self.input_y = tf.placeholder(shape=[self.batch_size, self.config.max_len], dtype=tf.int32,
                                      name=self.name+"_input_y")
        self.predict_y = tf.placeholder(shape=[self.batch_size, self.config.max_len], dtype=tf.int32,
                                        name=self.name+"_predict_y")
        self.emotion_labels = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_emotion_labels")
        self.emotion_word_mask = tf.placeholder(shape=[self.batch_size, self.num_total_word], dtype=tf.float32,
                                                name=self.name+"_emotion_word_mask")

        self.encode_fd_cell = self.rnn_cell()
        self.encode_bd_cell = self.rnn_cell()
        self.decode_cell = self.rnn_cell()

        self.entropy_loss, self.reg_loss = self.compute_loss()
        self.total_loss = self.entropy_loss + self.lambda_reg * self.reg_loss
        self.generation_words = self.generate()
        tf.summary.scalar("cross entropy loss", self.entropy_loss)
        tf.summary.scalar("regularization loss", self.reg_loss)
        tf.summary.scalar("total loss", self.total_loss)

        self.global_step = tf.Variable(0, name=self.name+"_global_step", trainable=False)
        self.lr = tf.Variable(self.config.optimizer.lr, dtype=tf.float32)
        self.lr_decay_op = self.lr.assign(self.lr * self.config.learning_rate_decay_factor)
        self.train = self.optimize(self.total_loss)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def activate(self, x):
        return self.config.activation.activate(x)

    def basic_rnn_cell(self):
        if self.config.use_lstm:
            return tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.nn.rnn_cell.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)

    def rnn_cell(self):
        single_cell = self.basic_rnn_cell
        if self.config.keep_prob < 1.0:
            def single_cell():
                return tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
        return cell

    def encode(self):
        input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_x)  # [batch, max_time, embedding size]

        initiate_state_forward = self.encode_fd_cell.zero_state(self.batch_size, dtype=tf.float32)
        initiate_state_backward = self.encode_bd_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.encode_fd_cell, self.encode_bd_cell, input_embeddings,
                                                          sequence_length=self.input_length,
                                                          initial_state_fw=initiate_state_forward,
                                                          initial_state_bw=initiate_state_backward,
                                                          dtype=tf.float32)
        output_fw_state, output_bw_state = states
        final_states = tf.add(output_fw_state, output_bw_state)  # [layer, cell, batch, hidden]
        split_states_outputs = tf.split(final_states, num_or_size_splits=2, axis=1)
        final_outputs = tf.reshape(split_states_outputs[1], [self.batch_size, self.hidden_size])
        return final_outputs, output_fw_state

    def post_attention(self, encode_hiddens, con_decode_hidden):
        """
        compute the post attention vector, means the context vector
        :param encode_hiddens: [batch, time_step, hidden]
        :param con_decode_hidden: [batch, hidden]
        :return: [batch, hidden]
        """
        post_w = tf.stack([self.post_trans_w for _ in range(self.batch_size)])  # [batch, hidden, 1]
        post_sim = tf.matmul(encode_hiddens, post_w)  # [batch, time, 1]
        single_decode = tf.matmul(con_decode_hidden, self.response_trans_w)  # [batch, 1]
        response_sim = tf.stack([single_decode for _ in range(self.config.max_len)], axis=1)  # [batch, time, 1]
        attention_scores = self.activate(post_sim + response_sim)  # [batch, time, 1]

        sfx_att_scores = tf.exp(attention_scores)
        sum_att_scores = tf.reduce_sum(sfx_att_scores, axis=1, keep_dims=True)
        sfx_att_scores = sfx_att_scores / sum_att_scores   # [batch, time, 1]
        post_att_vectors = tf.reduce_sum(sfx_att_scores * encode_hiddens, axis=1)  # [batch, hidden]
        return post_att_vectors

    def get_batch_emotion_words(self):
        this_batch_words = self.num_generic_word + self.emotion_vocab_size * self.emotion_labels  # [batch]
        batch_emotion_words = []
        for i in range(self.emotion_vocab_size):
            # [emotion word, batch]
            batch_emotion_words.append(this_batch_words)
            this_batch_words = this_batch_words + 1
        return tf.transpose(batch_emotion_words)

    def emotion_attention(self, last_encode_hiddens, con_decode_hiddens):
        """
        compute the emotion attention
        :param last_encode_hiddens: [batch, hidden]
        :param con_decode_hiddens: [batch, hidden]
        :return:
        """
        batch_emotion_words = self.get_batch_emotion_words()
        batch_emotion_embeddings = tf.nn.embedding_lookup(self.embeddings, batch_emotion_words)  # [batch, emotion, embedding]
        emotion_word_w = tf.stack([self.emotion_word_trans_w for _ in range(self.batch_size)], axis=0) # [batch, embedding, 1]
        emotion_word_sim = tf.matmul(batch_emotion_embeddings, emotion_word_w)  # [batch, emotion, 1]

        encode_hidden_single = tf.matmul(last_encode_hiddens, self.emotion_post_trans_w)  # [batch, 1]
        encode_hidden_sim = tf.stack([encode_hidden_single for _ in range(self.emotion_vocab_size)], axis=1) # [batch, emotion, 1]

        decode_hidden_single = tf.matmul(con_decode_hiddens, self.emotion_response_trans_w) # [batch, 1]
        decode_hidden_sim = tf.stack([decode_hidden_single for _ in range(self.emotion_vocab_size)], axis=1) # [batch, emotion, 1]

        emotion_attention_scores = self.activate(emotion_word_sim + encode_hidden_sim + decode_hidden_sim)  # [batch, emotion, 1]
        sfx_emotion_scores = tf.nn.softmax(emotion_attention_scores, dim=1)
        emotion_att_vectors = tf.reduce_sum(sfx_emotion_scores * batch_emotion_embeddings, axis=1)  # [batch, embedding]
        return emotion_att_vectors

    def get_generic_emotion_scores(self, predict_scores):
        """
        split generic and emotional predicting scores
        :param predict_scores: [batch, total words]
        :return:
        """
        # batch: [total words]
        split_predict_scores = [tf.reshape(score, [self.num_total_word])
                                for score in tf.split(predict_scores, num_or_size_splits=self.batch_size, axis=0)]
        emotion_words = self.get_batch_emotion_words()
        # batch: [emotion]
        split_emotion_words = [tf.reshape(words, [self.emotion_vocab_size])
                               for words in tf.split(emotion_words, num_or_size_splits=self.batch_size, axis=0)]
        generic_words_index = range(self.num_generic_word)

        generic_scores, emotion_scores = [], []
        for emotion_word, score in zip(split_emotion_words, split_predict_scores):
            this_generic_score = tf.nn.embedding_lookup(score, generic_words_index)
            this_emotion_score = tf.nn.embedding_lookup(score, emotion_words)

            generic_scores.append(this_generic_score)
            emotion_scores.append(this_emotion_score)
        batch_generic_scores = tf.reshape(generic_scores, [self.batch_size, self.num_generic_word])
        batch_emotion_scores = tf.reshape(emotion_scores, [self.batch_size, self.emotion_vocab_size])
        return batch_generic_scores, batch_emotion_scores

    def get_generic_word_mask(self):
        generic_mask = np.zeros([self.batch_size, self.num_total_word], dtype=float)  # unk: 0
        generic_mask[:, 1: self.num_generic_word] = 1.0
        return generic_mask

    def get_emotion_word_mask(self, this_emotion_labels):
        emotion_mask = np.zeros([self.batch_size, self.num_total_word], dtype=float)
        for i in range(self.batch_size):
            start_pos = self.num_generic_word + self.emotion_vocab_size * this_emotion_labels[i]
            end_pos = self.num_generic_word + self.emotion_vocab_size * (this_emotion_labels[i] + 1)
            emotion_mask[i, start_pos: end_pos] = 1.0
        return emotion_mask

    def get_emotion_mask(self, predict_alpha):
        """
        compute softmax alpha mask
        :param predict_alpha:
        :return:
        """
        """
        generic_alpha = 1.5 * predict_alpha
        emotion_alpha = 1 - generic_alpha
        predict_alpha = tf.exp(generic_alpha) / (tf.exp(generic_alpha) + tf.exp(emotion_alpha))
        """
        generic_mask_score = self.generic_word_mask * predict_alpha
        emotion_mask_score = self.emotion_word_mask * (1 - predict_alpha)
        mask_score = generic_mask_score + emotion_mask_score
        return mask_score

    def get_unscaled_predict_scores(self, predict_scores, predict_alpha):
        """
        compute final softmax_scores
        :param predict_scores: [batch, total words]
        :param predict_alpha: [batch, 1]
        :return:
        """
        emotion_mask = self.get_emotion_mask(predict_alpha)  # [batch, total]
        mask_predict_scores = predict_scores * emotion_mask
        # mask_predict_scores_smooth = mask_predict_scores + self.config.softmax_smooth  # [batch, total]
        mask_predict_scores_smooth = mask_predict_scores  # [batch, total]
        # sfx_scores = tf.nn.softmax(mask_predict_scores_smooth, dim=-1)
        return mask_predict_scores_smooth

    def decode(self, last_encode_hiddens, last_encode_state):
        """
        decoding process
        :param last_encode_hiddens: [batch, hidden]
        :param last_encode_state:
        :return:
        """
        input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_y)

        # initial_state = self.decode_cell.zero_state(self.batch_size, dtype=tf.float32)
        initial_state = last_encode_state
        outputs = []
        total_sfx_scores = []

        state = initial_state
        output = tf.zeros(shape=[self.batch_size, self.hidden_size], dtype=tf.float32)
        with tf.variable_scope(self.name + "_decoder"):
            for i in range(self.config.max_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                this_input_embeddings = input_embeddings[:, i, :]
                this_emotion_attention = self.emotion_attention(last_encode_hiddens, output)  # [batch, *]
                this_input_vectors = tf.concat([this_input_embeddings, this_emotion_attention], axis=1)
                (output, state) = self.decode_cell(this_input_vectors, state)
                outputs.append(output)

                predict_scores = tf.nn.relu(tf.matmul(output, self.sfx_w) + self.sfx_b)  # [batch, total]
                predict_alpha = tf.sigmoid(tf.matmul(output, self.sfx_score_w))  # [batch, 1]
                final_predict_scores_smooth = self.get_unscaled_predict_scores(predict_scores, predict_alpha)  # [batch, total]
                total_sfx_scores.append(final_predict_scores_smooth)

            total_sfx_scores = tf.transpose(total_sfx_scores, perm=[1, 0, 2])  # [batch, max len, total]
        return outputs, total_sfx_scores

    def get_generation_symbols(self, index, predict_scores=None):
        symbols = []
        if index == 0:
            for i in range(self.batch_size):
                symbols.append(self.start_id)
        else:
            symbols = tf.argmax(predict_scores, axis=1)
        return symbols

    def generate_response(self, last_encode_hiddens, last_encode_state):
        initial_state = last_encode_state
        state = initial_state
        output = tf.zeros(shape=[self.batch_size, self.hidden_size], dtype=tf.float32)
        this_emotion_attention = self.emotion_attention(last_encode_hiddens, output)  # [batch, *]

        response_words = []
        symbols = self.get_generation_symbols(0)
        response_words.append(symbols)
        with tf.variable_scope(self.name + "_decoder"):
            for i in range(self.config.max_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                this_input_embeddings = tf.nn.embedding_lookup(self.embeddings, symbols)  # [batch, embedding]
                this_input_vectors = tf.concat([this_input_embeddings, this_emotion_attention], axis=1)
                (output, state) = self.decode_cell(this_input_vectors, state)

                predict_scores = tf.sigmoid(tf.matmul(output, self.sfx_w) + self.sfx_b)  # [batch, total]
                predict_alpha = tf.sigmoid(tf.matmul(output, self.sfx_score_w))  # [batch, 1]
                final_predict_scores_smooth = self.get_unscaled_predict_scores(predict_scores,
                                                                               predict_alpha)  # [batch, total]
                symbols = self.get_generation_symbols(i + 1, final_predict_scores_smooth)
                response_words.append(symbols)
        return tf.transpose(response_words)

    def compute_loss(self):
        last_encoder_hiddens, last_encode_state = self.encode()

        # likelihood loss
        outputs, total_sfx_scores = self.decode(last_encoder_hiddens, last_encode_state)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.predict_y, logits=total_sfx_scores)
        loss_likelihood = tf.reduce_sum(cross_entropy) / self.batch_size

        # regularization loss
        normed_scores = tf.reduce_sum(tf.nn.softmax(total_sfx_scores, dim=-1), axis=1)  # [batch, total]
        split_norm_scores = tf.split(normed_scores, num_or_size_splits=self.batch_size)  # batch: [1, total]
        split_norm_scores = [tf.transpose(scores) for scores in split_norm_scores]  # batch: [total, 1]
        batch_norm_scores = [tf.reduce_sum(scores * self.embeddings, axis=0) for scores in split_norm_scores]
        expected_embeddings = tf.reshape(batch_norm_scores, [self.batch_size, self.embedding_size])
        post_dis = tf.matmul(expected_embeddings, self.reg_emotion_w) + self.reg_emotion_b  # [batch, emotion]
        true_dist = tf.one_hot(self.emotion_labels, depth=self.emotion_class + 1)  # [batch, emotion]
        loss_reg = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=true_dist, logits=post_dis)) \
                    / self.batch_size
        return loss_likelihood, loss_reg

    def optimize(self, loss):
        optimizer = self.config.optimizer.get_optimizer()
        trainer = optimizer.minimize(loss)
        return trainer

    def generate(self):
        last_encoder_hiddens, last_encode_state = self.encode()
        generate_words = self.generate_response(last_encoder_hiddens, last_encode_state)  # [batch, max len]
        return generate_words

    def train_step(self, this_input_x, this_post_len, this_input_y, this_predict_y, this_emotion_labels,
                   this_emotion_mask):
        input_feed = {self.input_x: this_input_x,
                      self.input_length: this_post_len,
                      self.input_y: this_input_y,
                      self.predict_y: this_predict_y,
                      self.emotion_labels: this_emotion_labels,
                      self.emotion_word_mask: this_emotion_mask}

        output_feed = [self.train, self.entropy_loss, self.reg_loss, self.total_loss]
        results = self.session.run(output_feed, input_feed)
        _, entropy_loss, reg_loss, total_loss = results
        return entropy_loss, reg_loss, total_loss

    def generate_step(self, this_input_x, this_post_len, this_emotion_labels, this_emotion_mask):
        input_feed = {self.input_x: this_input_x,
                      self.input_length: this_post_len,
                      self.emotion_labels: this_emotion_labels,
                      self.emotion_word_mask: this_emotion_mask}

        output_feed = [self.generation_words, self.embeddings]
        results = self.session.run(output_feed, input_feed)
        words, embeddings = results[0], results[1]
        return words, embeddings

    def get_batch(self, input_post, input_post_length, input_response, predict_response, emotion_labels, index):
        this_input_x = input_post[self.batch_size * index: self.batch_size * (index + 1)]
        this_input_x_len = input_post_length[self.batch_size * index: self.batch_size * (index + 1)]
        this_input_y = input_response[self.batch_size * index: self.batch_size * (index + 1)]
        this_predict_y = predict_response[self.batch_size * index: self.batch_size * (index + 1)]
        this_emotion_labels = emotion_labels[self.batch_size * index: self.batch_size * (index + 1)]
        this_emotion_mask = self.get_emotion_word_mask(this_emotion_labels)
        return this_input_x, this_input_x_len, this_input_y, this_predict_y, this_emotion_labels, this_emotion_mask

    def get_test_batch(self, input_post, input_post_length, emotion_labels, index):
        this_input_x = input_post[self.batch_size * index: self.batch_size * (index + 1)]
        this_input_x_len = input_post_length[self.batch_size * index: self.batch_size * (index + 1)]
        this_emotion_labels = emotion_labels[self.batch_size * index: self.batch_size * (index + 1)]
        this_emotion_mask = self.get_emotion_word_mask(this_emotion_labels)
        return this_input_x, this_input_x_len, this_emotion_labels, this_emotion_mask


























