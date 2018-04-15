
# RNN model Class

import tensorflow as tf
import numpy as np
tf.set_random_seed(77)
layers = tf.contrib.layers

class RNN(object):
    def __init__(self,
                 song_length,
                 batch_size=2,
                 mode='train'):

        # mode 'train' or 'test'
        self.mode = mode

        # set index <-> data
        # melody [0, 1 ~ 36, 50] -- size 38
        # rhythm [0, 1, 2, 3, 4, 6, 8, 12, 16] -- size 9
        self.melody_sample = list(range(1, 37))
        self.melody_sample.append(50)  # rest
        self.melody_sample.append(0)  # for test
        self.idx2char = list(set(self.melody_sample))
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

        # set hyperparameter
        self.input_size = len(self.char2idx)  # 37
        self.hidden_size = 128
        self.output_size = len(self.char2idx)  # 37
        self.batch_size = batch_size
        self.sequence_length = song_length - 1
        self.learning_rate = 0.01

        # placeholder
        self.X = tf.placeholder(tf.int32, [None, self.sequence_length], name='x_data')
        self.Y = tf.placeholder(tf.int32, [None, self.sequence_length], name='y_data')

    def data2idx(self, data):
        x_data = []
        y_data = []
        for d in data:
            train_data = [self.char2idx[i] for i in d]
            x_data.append(train_data[:-1])
            y_data.append(train_data[1:])

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data

    def build(self):

        # one_hot : [sequence_length, input_size]
        self.x_one_hot = tf.one_hot(self.X, self.input_size)
        print (self.x_one_hot.shape)

        with tf.variable_scope("RNN"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                state_is_tuple=True)

            # defining initial state
            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            if (self.mode=='train'):
                self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            elif (self.mode=='test'):
                self.initial_state = cell.zero_state(1, dtype=tf.float32)

            '''
            ########### test #############
            output, states = cell(self.x_one_hot[:, 0, :], self.initial_state)
            X_for_fc = tf.reshape(output, [-1, self.hidden_size])
            with tf.variable_scope("output_fully") as scope:
                output_fully = layers.fully_connected(inputs=X_for_fc,
                                                      num_outputs=self.output_size,
                                                      activation_fn=None)
            prev_output = output_fully
            self.outputs = []
            self.outputs.append(output_fully)

            for step in range(self.sequence_length - 1):

                if (self.mode == 'train'):
                    output, states = cell(self.x_one_hot[:, step + 1, :], states)

                elif (self.mode == 'test'):
                    output, states = cell(prev_output, states)

                X_for_fc = tf.reshape(output, [-1, self.hidden_size])
                with tf.variable_scope("output_fully") as scope:
                    scope.reuse_variables()
                    output_fully = layers.fully_connected(inputs=X_for_fc,
                                                          num_outputs=self.output_size,
                                                          activation_fn=None)

                self.outputs.append(output_fully)  # [squence_length, batch_size, output_size]
                prev_output = output_fully

            self.outputs = tf.transpose(self.outputs, perm=[1, 0, 2])  # [batch_size, squence_length, output_size]

            ########### test #############
            '''

            ########### dynamic ##########
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.x_one_hot,
                                               initial_state=self.initial_state)


            # output size : [batch_size, seqence_length, hidden_size]
            X_for_fc = tf.reshape(outputs, [-1, self.hidden_size])
            outputs = layers.fully_connected(inputs=X_for_fc,
                                             num_outputs=self.output_size,
                                             activation_fn=None)

            # reshape outputs for sequence_loss
            self.outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_size])

            ########### dynamic ##########


            self.FC_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RNN')

        with tf.variable_scope("loss"):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                             targets=self.Y,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.FC_vars)
            tf.summary.scalar("loss", self.loss)

        self.prediction = tf.argmax(self.outputs, axis=2)

        print('complete model build.')