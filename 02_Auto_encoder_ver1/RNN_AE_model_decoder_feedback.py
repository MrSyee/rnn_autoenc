
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
        self.input_size = len(util.pitch_sample)
        self.hidden_size = 128
        self.output_size = len(util.pitch_sample)
        self.batch_size = batch_size
        self.sequence_length = song_length
        self.learning_rate = 0.01

        # cell
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                     state_is_tuple=False)
        self.dec_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                     state_is_tuple=False)

        # placeholder
        self.Enc_input = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.Dec_state = tf.placeholder(tf.float32, [self.batch_size, self.enc_cell.state_size])

    def encoder(self, scopename):
        self.enc_scope = "enc_{}".format(scopename)
        # one_hot : [sequence_length, input_size]
        self.x_one_hot = tf.one_hot(self.Enc_input, self.input_size)

        with tf.variable_scope(self.enc_scope):

            # defining initial state
            self.initial_state = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)

            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                       inputs=self.x_one_hot,
                                                       initial_state=self.initial_state)

        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.enc_scope)

        return enc_state


    def decoder(self, scopename, istrain=False):
        self.dec_scope = "dec_{}".format(scopename)

        with tf.variable_scope(self.dec_scope):

            if istrain:
                # ex. [1,2,3,4] -> [0,4,3,2] => decoder input
                dec_input = tf.slice(self.Enc_input, [0, 1], [self.batch_size, self.sequence_length - 1])
                dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(dec_input, [1])], 1)

                dec_input_onehot = tf.one_hot(dec_input, self.input_size)
                # dec_input_onehot:
                # [batch_size, sequence_length, one-hot size] -> [sequence_length, batch_size, one-hot size]
                dec_input_onehot = tf.transpose(dec_input_onehot, perm=[1, 0, 2])

                output, state = self.dec_cell(dec_input_onehot[0], self.dec_state)
            else:
                # input start signal 'Zero'
                zero_one_hot = tf.one_hot(tf.zeros([self.batch_size], tf.int32), self.input_size)
                output, state = self.dec_cell(zero_one_hot, self.Dec_state)

            # output : [batch_size, hidden_size]
            with tf.variable_scope("output_fully") as scope:
                outputs = layers.fully_connected(inputs=output,
                                                 num_outputs=self.output_size,
                                                 activation_fn=None)
            # [batch_size, one-hot size]
            self.dec_state = state
            dec_cell_input = outputs
            dec_cell_input = tf.argmax(dec_cell_input, axis=1)
            dec_cell_input = tf.one_hot(dec_cell_input, self.input_size)

            # [sequence_size, batch_size, one-hot size]
            dec_output = []
            dec_output.append(outputs)

            for i in range(self.sequence_length-1):
                if istrain:
                    output, state = self.dec_cell(dec_input_onehot[i + 1], self.dec_state)
                else:
                    output, state = self.dec_cell(dec_cell_input, self.dec_state)

                with tf.variable_scope("output_fully") as scope:
                    scope.reuse_variables()
                    outputs = layers.fully_connected(inputs=output,
                                                     num_outputs=self.output_size,
                                                     activation_fn=None)
                self.dec_state = state
                dec_output.append(outputs)
                dec_cell_input = outputs
                dec_cell_input = tf.argmax(dec_cell_input, axis=1)
                dec_cell_input = tf.one_hot(dec_cell_input, self.input_size)

            # reshape outputs for sequence_loss
            #outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_size])
            # [batch_size, sequence_size, one-hot size]
            outputs = tf.transpose(dec_output, perm=[1,0,2])
            self.outputs = tf.reverse(outputs, [1])
            self.prediction = tf.argmax(self.outputs, axis=2)

        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dec_scope)

    def build(self, scopename):
        print("Start model build...")
        self.loss_scope = "loss_{}".format(scopename)

        # one_hot : [sequence_length, input_size]
        self.x_one_hot = tf.one_hot(self.Enc_input, self.input_size)

        # encoder build
        self.dec_state = self.encoder(scopename)

        # decoder build
        self.decoder(scopename, istrain=True)

        # decoder loss
        with tf.variable_scope(self.loss_scope):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                             targets=self.Enc_input,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=[self.enc_vars, self.dec_vars])

        # summary for tensorboard graph
        self._create_summaries()

        print('complete model build.')

    def _create_summaries(self):
        with tf.variable_scope('summaries'):
            summ_loss = tf.summary.scalar(self.loss_scope, self.loss)

            self.summary_op = tf.summary.merge([summ_loss])