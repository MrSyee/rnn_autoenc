
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
from datetime import datetime

from RNN_AE_model_decoder_feedback_melody import LSTMAutoEnc
import utils

now = datetime.now()
util = utils.Util()

NOWTIME = now.strftime("%Y%m%d-%H%M%S")

def train(song, model):
    '''
    train the model using data
    :param song: Dict, Song data
        model: class, NN Model
        mode: string, "pitches" or "durations"
    :return:
    '''
    print ('Start Train')

    # Train the RNN model
    step = 1000

    model.build()

    # make char2idx
    p_char2idx = util.getchar2idx(mode="pitch")
    d_char2idx = util.getchar2idx(mode="duration")

    p_enc_saver = tf.train.Saver(var_list=model.p_enc_vars)
    d_enc_saver = tf.train.Saver(var_list=model.d_enc_vars)
    p_dec_saver = tf.train.Saver(var_list=model.p_dec_vars)
    d_dec_saver = tf.train.Saver(var_list=model.d_dec_vars)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + NOWTIME, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            pitch_data = util.data2idx([song['pitches']], p_char2idx)
            pitch_data = np.reshape(pitch_data, [model.batch_size, pitch_data.shape[1]])
            duration_data = util.data2idx([song['durations']], d_char2idx)
            duration_data = np.reshape(duration_data, [model.batch_size, duration_data.shape[1]])

            p_loss, d_loss, _, _ = sess.run([model.p_loss, model.d_loss, model.p_train, model.p_train],
                                            feed_dict={model.p_input: pitch_data,
                                                       model.d_input: duration_data})
            p, d = sess.run([model.p_prediction, model.d_prediction],
                            feed_dict={model.p_input: pitch_data,
                                       model.d_input: duration_data})

            summary = sess.run(model.summary_op, feed_dict={model.p_input: pitch_data,
                                                            model.d_input: duration_data})

            writer.add_summary(summary, i)

            # result_str = [model.idx2char[c] for c in np.squeeze(result)]

            print("{:4d}  p_loss: {:.5f}  d_loss: {:.5f}".format(i+1, p_loss, d_loss))

        p_enc_saver.save(sess, "./save/" + NOWTIME + "/pitch_enc_model.ckpt")
        d_enc_saver.save(sess, "./save/" + NOWTIME + "/duration_enc_model.ckpt")
        p_dec_saver.save(sess, "./save/" + NOWTIME + "/pitch_dec_model.ckpt")
        d_dec_saver.save(sess, "./save/" + NOWTIME + "/duration_dec_model.ckpt")


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

    # net
    net = LSTMAutoEnc(song_length=trained_song['length'],
                      batch_size=1,
                      mode='train')

    # train NN
    train(trained_song, net)

if __name__ == '__main__':
    tf.app.run()

