
# RNN Auto Encoder model Class

import tensorflow as tf
import numpy as np
import utils

tf.set_random_seed(77)
layers = tf.contrib.layers
util = utils.Util()

class LSTMAutoEnc(object):
    def __init__(self,
                 song_length,
                 batch_size=1,
                 mode='train'):

        # mode 'train' or 'test'
        self.mode = mode

        # set hyperparameter
        self.pitch_size = len(util.pitch_sample)
        self.duration_size = len(util.duration_sample)
        self.output_size = self.pitch_size + self.duration_size
        self.hidden_size = 128
        self.batch_size = batch_size
        self.sequence_length = song_length
        self.learning_rate = 0.01

        # cell
        self.p_enc_cell= tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                      state_is_tuple=False)
        self.d_enc_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                       state_is_tuple=False)
        self.p_dec_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                       state_is_tuple=False)
        self.d_dec_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                       state_is_tuple=False)

        # placeholder
        self.p_input = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.d_input = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.p_Dec_state = tf.placeholder(tf.float32, [self.batch_size, self.p_enc_cell.state_size])
        self.d_Dec_state = tf.placeholder(tf.float32, [self.batch_size, self.d_enc_cell.state_size])

    def encoder(self):
        self.p_enc_scope = "enc_pitch"
        self.d_enc_scope = "enc_duration"
        # one_hot : [batch_size, sequence_length, input_size]
        self.p_one_hot = tf.one_hot(self.p_input, self.pitch_size)
        self.d_one_hot = tf.one_hot(self.d_input, self.duration_size)
        self.x_one_hot = tf.concat([self.p_one_hot, self.d_one_hot], axis=2)
        # x_one_hot = [batch_size, sequnece_length, pitch+duration_size]

        with tf.variable_scope(self.p_enc_scope):
            # defining initial state
            initial_state = self.p_enc_cell.zero_state(self.batch_size, dtype=tf.float32)
            enc_outputs, p_enc_state = tf.nn.dynamic_rnn(cell=self.p_enc_cell,
                                               inputs=self.x_one_hot,
                                               initial_state=initial_state)
        with tf.variable_scope(self.d_enc_scope):
            # defining initial state
            initial_state = self.d_enc_cell.zero_state(self.batch_size, dtype=tf.float32)
            enc_outputs, d_enc_state = tf.nn.dynamic_rnn(cell=self.d_enc_cell,
                                               inputs=self.x_one_hot,
                                               initial_state=initial_state)

        self.p_enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.p_enc_scope)
        self.d_enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.d_enc_scope)

        return p_enc_state, d_enc_state

    def decoder_op(self, scopename, inputs, state, cell, output_size, reuse):
        lstm_scope = "dec_{}".format(scopename)
        fully_scope = "output_fully_{}".format(scopename)

        with tf.variable_scope(lstm_scope) as scope:
            if reuse:
                scope.reuse_variables()
            # Use LSTM Cell
            output, state = cell(inputs, state)

            # output : [batch_size, hidden_size]
            with tf.variable_scope(fully_scope) as scope:
                if reuse:
                    scope.reuse_variables()
                output = layers.fully_connected(inputs=output,
                                                num_outputs=output_size,
                                                activation_fn=None)
                return output, state

    def decoder(self, istrain=False):
        if istrain:
            # ex. [1,2,3,4] -> [0,4,3,2] => decoder input
            p_dec_input = tf.slice(self.p_input, [0, 1], [self.batch_size, self.sequence_length - 1])
            p_dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(p_dec_input, [1])], 1)
            p_dec_input_onehot = tf.one_hot(p_dec_input, self.pitch_size)

            d_dec_input = tf.slice(self.d_input, [0, 1], [self.batch_size, self.sequence_length - 1])
            d_dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(d_dec_input, [1])], 1)
            d_dec_input_onehot = tf.one_hot(d_dec_input, self.duration_size)

            dec_input_onehot = tf.concat([p_dec_input_onehot, d_dec_input_onehot], axis=2)

            # dec_input_onehot: [batch_size, sequence_length, one-hot size] -> [sequence_length, batch_size, one-hot size]
            dec_input_onehot = tf.transpose(dec_input_onehot, perm=[1, 0, 2])

            # output: [batch_size, pitch or duration size]
            p_output, p_state = self.decoder_op('pitch', dec_input_onehot[0], self.p_Dec_state,
                                                self.p_dec_cell, self.pitch_size, reuse=False)
            d_output, d_state = self.decoder_op('duration', dec_input_onehot[0], self.d_Dec_state,
                                                self.d_dec_cell, self.duration_size, reuse=False)

        else:
            # input start signal 'Zero'
            zero_one_hot_p = tf.one_hot(tf.zeros([self.batch_size, 1], tf.int32), self.pitch_size)
            zero_one_hot_d = tf.one_hot(tf.zeros([self.batch_size, 1], tf.int32), self.duration_size)
            zero_one_hot = tf.squeeze(tf.concat([zero_one_hot_p, zero_one_hot_d], axis=2), 0)

            # output: [batch_size, pitch or duration size]
            p_output, p_state = self.decoder_op('pitch', zero_one_hot, self.p_Dec_state,
                                                self.p_dec_cell, self.pitch_size, reuse=False)
            d_output, d_state = self.decoder_op('duration', zero_one_hot, self.d_Dec_state,
                                                self.d_dec_cell, self.duration_size, reuse=False)

        p_input = tf.argmax(p_output, axis=1)
        d_input = tf.argmax(d_output, axis=1)
        p_one_hot = tf.one_hot(p_input, self.pitch_size)
        d_one_hot = tf.one_hot(d_input, self.duration_size)
        dec_one_hot = tf.concat([p_one_hot, d_one_hot], axis=1)
        p_Dec_state = p_state
        d_Dec_state = d_state

        # [sequence_size, batch_size, one-hot size]
        pitch_output = []
        pitch_output.append(p_output)
        duration_output = []
        duration_output.append(d_output)

        for i in range(self.sequence_length-1):
            if istrain:
                p_output, p_state = self.decoder_op('pitch', dec_input_onehot[i+1], p_Dec_state,
                                                    self.p_dec_cell, self.pitch_size, reuse=True)
                d_output, d_state = self.decoder_op('duration', dec_input_onehot[i+1], d_Dec_state,
                                                    self.d_dec_cell, self.duration_size, reuse=True)
            else:
                p_output, p_state = self.decoder_op('pitch', dec_one_hot, p_Dec_state,
                                                    self.p_dec_cell, self.pitch_size, reuse=True)
                d_output, d_state = self.decoder_op('duration', dec_one_hot, d_Dec_state,
                                                    self.d_dec_cell, self.duration_size, reuse=True)

            p_input = tf.argmax(p_output, axis=1)
            d_input = tf.argmax(d_output, axis=1)
            p_one_hot = tf.one_hot(p_input, self.pitch_size)
            d_one_hot = tf.one_hot(d_input, self.duration_size)
            dec_one_hot = tf.concat([p_one_hot, d_one_hot], axis=1)
            p_Dec_state = p_state
            d_Dec_state = d_state

            # [sequence_size, batch_size, one-hot size]
            pitch_output.append(p_output)
            duration_output.append(d_output)

            # [batch_size, sequence_size, one-hot size]
            self.p_outputs = tf.transpose(pitch_output, perm=[1, 0, 2])
            #self.p_outputs = tf.reverse(p_outputs, [1])
            self.d_outputs = tf.transpose(duration_output, perm=[1, 0, 2])
            #self.d_outputs = tf.reverse(d_outputs, [1])
            self.p_prediction, self.d_prediction = tf.argmax(self.p_outputs, axis=2), tf.argmax(self.d_outputs, axis=2)

        self.p_dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dec_pitch")
        self.d_dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dec_duration")

    def build(self):
        print("Start model build...")
        self.p_loss_scope = "loss_pitch"
        self.d_loss_scope = "loss_duration"

        # encoder build
        self.p_Dec_state, self.d_Dec_state = self.encoder()

        # decoder build
        self.decoder(istrain=True)

        # pitch decoder loss
        with tf.variable_scope(self.p_loss_scope):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.p_outputs,
                                                             targets=self.p_input,
                                                             weights=weights)
            self.p_loss = tf.reduce_mean(sequence_loss)
            self.p_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.p_loss,
                                                                                             var_list=[self.p_enc_vars, self.p_dec_vars])
        # duration decoder loss
        with tf.variable_scope(self.d_loss_scope):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.d_outputs,
                                                             targets=self.d_input,
                                                             weights=weights)
            self.d_loss = tf.reduce_mean(sequence_loss)
            self.d_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss,
                                                                                             var_list=[self.d_enc_vars, self.d_dec_vars])

        # summary for tensorboard graph
        self._create_summaries()

        print('complete model build.')

    def _create_summaries(self):
        with tf.variable_scope('summaries'):
            summ_p_loss = tf.summary.scalar(self.p_loss_scope, self.p_loss)
            summ_d_loss = tf.summary.scalar(self.d_loss_scope, self.d_loss)

            self.summary_op = tf.summary.merge([summ_p_loss, summ_d_loss])