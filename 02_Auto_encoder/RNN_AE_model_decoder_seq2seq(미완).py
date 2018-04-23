
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

        print("batch_size : ", self.batch_size)
        print("enc_cell_state_size : ", self.enc_cell.state_size)
        print("zero_enc_state_size : ", np.shape(init))

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

            self.dec_input_onehot = tf.one_hot(dec_input, self.input_size)

            def loop_fn_initial():
                initial_elements_finished = (0 >= self.sequence_length)  # all False at the initial step
                initial_input = eos_step_embedded
                initial_cell_state = self.Dec_state
                initial_cell_output = None
                initial_loop_state = None  # we don't need to pass any additional information
                return (initial_elements_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)

            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
                def get_next_input():
                    with tf.variable_scope("FC"):
                        # output_logits = tf.add(tf.matmul(previous_output, W), b)
                        # output size : [batch_size, hidden_size]
                        # hidden_size -> one-hot size
                        outputs = layers.fully_connected(inputs=previous_output,
                                                         num_outputs=self.output_size,
                                                         activation_fn=None)
                        # output size : [batch_size, one-hot size]
                        # outputs = tf.reverse(outputs, [1])

                    return outputs

                elements_finished = (time >= self.sequence_length)  # this operation produces boolean tensor of [batch_size]
                # defining if corresponding sequence has ended

                # finished = tf.reduce_all(elements_finished)  # -> boolean scalar
                # input = tf.cond(finished, lambda: pad_step_embedded, get_next_input())
                input = get_next_input()
                state = previous_state
                output = previous_output
                loop_state = None

                return (elements_finished,
                        input,
                        state,
                        output,
                        loop_state)

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:  # time == 0
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.dec_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()

            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
            with tf.variable_scope("FC"):
                decoder_logits_flat = layers.fully_connected(inputs=decoder_outputs_flat,
                                                 num_outputs=self.output_size,
                                                 activation_fn=None)
            decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.sequence_length))

            self.prediction = tf.argmax(decoder_logits, 2)

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
            scope.reuse_variables()
            # defining initial state
            self.initial_state = self.enc_cell.zero_state(self.batch_size, dtype=tf.float32)

            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                               inputs=self.x_one_hot,
                                               initial_state=self.initial_state)
            # output size : [batch_size, seqence_length, hidden_size]


        '''
        with tf.variable_scope("decoder"):

            self.dec_state = enc_state
            print ("state_size : ", np.shape(self.dec_state) )

            # ex. [1,2,3,4] -> [0,4,3,2] => decoder input
            dec_input = tf.slice(self.Dec_input, [0,1], [self.batch_size, self.sequence_length-1])
            dec_input = tf.concat([tf.zeros([self.batch_size, 1], tf.int32), tf.reverse(dec_input, [1])], 1)

            self.dec_input_onehot = tf.one_hot(dec_input, self.input_size)

            dec_output, dec_state = tf.nn.dynamic_rnn(cell=self.dec_cell,
                                                      inputs=self.dec_input_onehot,
                                                      initial_state=self.dec_state)
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

            self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        '''

        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

        self.decoder()

        with tf.variable_scope("loss"):
            weights = tf.ones([self.batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                             targets=self.Enc_input,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=[self.enc_vars, self.dec_vars])
            tf.summary.scalar("loss", self.loss)


        print('complete model build.')