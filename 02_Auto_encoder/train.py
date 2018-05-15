
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
from datetime import datetime

import RNN_AE_model_decoder_feedback as rnn_AE
import utils

now = datetime.now()
util = utils.Util()

NOWTIME = now.strftime("%Y%m%d-%H%M%S")

def train(trained_data, model, mode):
    '''
    train the model using data
    :param trained_song: list, Song data
        model: class, NN Model
        mode: string, "pitches" or "durations"
    :return:
    '''
    print ('Start Train : {}'.format(mode))

    # Train the RNN model
    step = 1000

    model.build(scopename=mode)

    # make char2idx
    char2idx = util.getchar2idx(mode=mode)

    enc_saver = tf.train.Saver(var_list=model.enc_vars)
    dec_saver = tf.train.Saver(var_list=model.dec_vars)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + NOWTIME + "/{}".format(mode), graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            x_data = util.data2idx(trained_data, char2idx)
            x_data = np.reshape(x_data, [model.batch_size, x_data.shape[1]])

            loss_val, _ = sess.run([model.loss, model.train], feed_dict={model.Enc_input: x_data,
                                                                         model.Dec_input: x_data})
            result = sess.run(model.prediction, feed_dict={model.Enc_input: x_data,
                                                           model.Dec_input: x_data})

            summary = sess.run(model.summary_op, feed_dict={model.Enc_input: x_data,
                                                      model.Dec_input: x_data})
            writer.add_summary(summary, i)

            #result_str = [model.idx2char[c] for c in np.squeeze(result)]

            print("{:4d}  loss: {:.5f}".format(i, loss_val))

        enc_saver.save(sess, "./save/" + NOWTIME + "/enc_{}_model.ckpt".format(mode))
        dec_saver.save(sess, "./save/" + NOWTIME + "/dec_{}_model.ckpt".format(mode))


def main(_):
    # load all midi file
    all_song = util.all_song

    # load one midi file
    filename = 'test.mid'
    trained_song = util.get_one_song(filename)
    print("One song load : {}".format(filename))
    print("name : ", trained_song['name'])
    print("length : ", trained_song['length'])
    print("pitches : ", trained_song['pitches'])
    print("durations : ", trained_song['durations'])

    # pitch net
    pitch_net = rnn_AE.LSTMAutoEnc(song_length=trained_song['length'],
                               batch_size=1,
                               mode='train')
    # duration net
    duration_net = rnn_AE.LSTMAutoEnc(song_length=trained_song['length'],
                                batch_size=1,
                                mode='train')
    # train NN
    train([trained_song['pitches']], pitch_net, mode='pitch')
    train([trained_song['durations']], duration_net, mode='duration')

if __name__ == '__main__':
    tf.app.run()
