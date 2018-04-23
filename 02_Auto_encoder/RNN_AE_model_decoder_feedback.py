
# RNN Auto Encoder model Class

import tensorflow as tf
import numpy as np
tf.set_random_seed(77)
layers = tf.contrib.layers

class LSTMAutoEnc(object):
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
        self.input_size = len(self.char2idx)
        self.hidden_size = 128
        self.output_size = len(self.char2idx)
        self.batch_size = batch_size
        self.sequence_length = song_length
        self.learning_rate = 0.01

        # cell
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                            state_is_tuple=False)
        self.dec_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                            state_is_tuple=False)

        # placeholder
        self.Enc_input = tf.placeholder(tf.int32, [None, self.sequence_length], name='enc_input')
        self.Dec_input = tf.placeholder(tf.int32, [None, self.sequence_length], name='dec_input')
        self.Dec_state = tf.placeholder(tf.float32, [self.batch_size, self.enc_cell.state_size], name='dec_state')

        init = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)

    def data2idx(self, data):
        x_data = []
        for d in data:
            train_data = [self.char2idx[i] for i in d]
            x_data.append(train_data[:])
        x_data = np.array(x_data)

        return x_data

    def encoder(self):
        # one_hot : [sequence_length, input_size]
        self.x_one_hot = tf.one_hot(self.Enc_input, self.input_size)

        with tf.variable_scope("encoder"):

            # defining initial state
            self.initial_state = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)

            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                               inputs=self.x_one_hot,
                                               initial_state=self.initial_state)

        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

        return enc_state


    def decoder(self):
        with tf.variable_scope("decoder"):
            # ex. [1,2,3,4] -> [0,4,3,2] => decoder input
            dec_input = tf.slice(self.Dec_input, [0, 1], [self.batch_size, self.sequence_length - 1])
            dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(dec_input, [1])], 1)

            dec_input_onehot = tf.one_hot(dec_input, self.input_size)
            # [batch_size, sequence_length, one-hot size] -> [sequence_length, batch_size, one-hot size]
            dec_input_onehot = tf.transpose(dec_input_onehot, perm=[1,0,2])

            output, state = self.dec_cell(dec_input_onehot[0], self.Dec_state)

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

            '''
            dec_output, dec_state = tf.nn.dynamic_rnn(cell=self.dec_cell,
                                                      inputs=self.dec_input_onehot,
                                                      initial_state=self.Dec_state)
            # output size : [batch_size, sueqence_length, hidden_size]
            dec_output_for_FC = tf.reshape(dec_output, [-1, self.hidden_size])

            # hidden_size -> one-hot size
            outputs = layers.fully_connected(inputs=dec_output_for_FC,
                                             num_outputs=self.output_size,
                                             activation_fn=None)
            # reshape outputs for sequence_loss
            outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_size])
            self.outputs = tf.reverse(outputs, [1])
            self.prediction = tf.argmax(self.outputs, axis=2)
            '''

        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def build(self):

        # one_hot : [sequence_length, input_size]
        self.x_one_hot = tf.one_hot(self.Enc_input, self.input_size)

        with tf.variable_scope("encoder") as scope:
            #scope.reuse_variables()
            # defining initial state
            self.initial_state = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)

            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                               inputs=self.x_one_hot,
                                               initial_state=self.initial_state)
            # output size : [batch_size, seqence_length, hidden_size]


        with tf.variable_scope("decoder"):

            self.dec_state = enc_state
            print ("state_size : ", np.shape(self.dec_state) )

            # ex. [1,2,3,4] -> [0,4,3,2] => decoder input
            dec_input = tf.slice(self.Dec_input, [0,1], [self.batch_size, self.sequence_length-1])
            dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(dec_input, [1])], 1)

            dec_input_onehot = tf.one_hot(dec_input, self.input_size)
            # dec_input_onehot = [batch_size, sequence_length, one-hot size] -> [sequence_length, batch_size, one-hot size]
            dec_input_onehot = tf.transpose(dec_input_onehot, perm=[1,0,2])

            output, state = self.dec_cell(dec_input_onehot[0], self.dec_state)

            # output : [batch_size, hidden_size]
            with tf.variable_scope("output_fully") as scope:
                outputs = layers.fully_connected(inputs=output,
                                                 num_outputs=self.output_size,
                                                 activation_fn=None)
            # [batch_size, one-hot size]
            self.dec_state = state

            # [sequence_size, batch_size, one-hot size]
            dec_output = []
            dec_output.append(outputs)

            for i in range(self.sequence_length-1):
                output, state = self.dec_cell(dec_input_onehot[i+1], self.dec_state)

                with tf.variable_scope("output_fully") as scope:
                    scope.reuse_variables()
                    outputs = layers.fully_connected(inputs=output,
                                                     num_outputs=self.output_size,
                                                     activation_fn=None)
                self.dec_state = state
                dec_output.append(outputs)


            '''
            dec_output, dec_state = tf.nn.dynamic_rnn(cell=self.dec_cell,
                                                      inputs=self.dec_input_onehot,
                                                      initial_state=self.dec_state)

            # output size : [batch_size, sueqence_length, hidden_size]
            dec_output_for_FC = tf.reshape(dec_output, [-1, self.hidden_size])

            # hidden_size -> one-hot size
            outputs = layers.fully_connected(inputs=dec_output_for_FC,
                                             num_outputs=self.output_size,
                                             activation_fn=None)
            '''
            # reshape outputs for sequence_loss
            #outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_size])
            # [batch_size, sequence_size, one-hot size]
            outputs = tf.transpose(dec_output, perm=[1,0,2])
            print (np.shape(outputs))
            self.outputs = tf.reverse(outputs, [1])
            self.prediction = tf.argmax(self.outputs, axis=2)



        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')


        with tf.variable_scope("loss"):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                             targets=self.Enc_input,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=[self.enc_vars, self.dec_vars])
            tf.summary.scalar("loss", self.loss)


        print('complete model build.')