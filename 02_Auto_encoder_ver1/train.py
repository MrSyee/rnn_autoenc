
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
from datetime import datetime

import RNN_AE_model_decoder_feedback as rnn_AE
import utils

util = utils.Util()
now = datetime.now()

NOWTIME = now.strftime("%Y%m%d-%H%M%S")


def train(trained_data, model, mode):
    '''
    train the model using data
    :param trained_song: list, Song data
        model: class, NN Model
        mode: string, "pitches" or "durations"
    :return:
    '''
    print('Start Train : {}'.format(mode))

    # Train the RNN model
    step = 1000

    model.build(scopename=mode)

    # make char2idx
    char2idx = util.getchar2idx(mode=mode)

    enc_saver = tf.train.Saver(var_list=model.enc_vars)
    dec_saver = tf.train.Saver(var_list=model.dec_vars)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + NOWTIME, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            x_data = util.data2idx(trained_data, char2idx)
            x_data = np.reshape(x_data, [model.batch_size, x_data.shape[1]])

            loss_val, _ = sess.run([model.loss, model.train], feed_dict={model.Enc_input: x_data})
            result = sess.run(model.prediction, feed_dict={model.Enc_input: x_data})

            summary = sess.run(model.summary_op, feed_dict={model.Enc_input: x_data})
            writer.add_summary(summary, i)

            #result_str = [model.idx2char[c] for c in np.squeeze(result)]

            print("{:4d}  loss: {:.5f}".format(i, loss_val))

        enc_saver.save(sess, "./save/" + NOWTIME + "/enc_{}_model.ckpt".format(mode))
        dec_saver.save(sess, "./save/" + NOWTIME + "/dec_{}_model.ckpt".format(mode))


def main(_):
    ### Song setting ###
    # load one midi file
    filename = 'test.mid'
    trained_song = util.get_one_song(filename)
    print("One song load : {}".format(filename))
    '''
    print("name : ", trained_song['name'])
    print("length : ", trained_song['length'])
    print("pitches : ", trained_song['pitches'])
    print("durations : ", trained_song['durations'])
    '''
    # load all midi file
    all_songs = util.all_song

    songs = all_songs
    print("Load {} Songs...".format(len(songs)))
    songs_len = []
    songs_pitches = []
    songs_durations = []
    for song in songs:
        print("name : ", song['name'])
        print("length : ", song['length'])
        print("pitches : ", song['pitches'])
        print("durations : ", song['durations'])
        print("")

        songs_len.append(song['length'])
        songs_pitches.append(song['pitches'])
        songs_durations.append(song['durations'])

    # 여러 곡의 길이를 제일 짧은 곡에 맞춘다.
    for i in range(len(songs_pitches)):
        if len(songs_pitches[i]) > min(songs_len):
            songs_pitches[i] = songs_pitches[i][:min(songs_len)]
            songs_durations[i] = songs_durations[i][:min(songs_len)]

    ### Train setting ###
    num_songs = len(songs)
    num_melody = min(songs_len)

    # pitch net
    pitch_net = rnn_AE.LSTMAutoEnc(song_length=num_melody,
                                   batch_size=num_songs,
                                   mode='train')
    # duration net
    duration_net = rnn_AE.LSTMAutoEnc(song_length=num_melody,
                                      batch_size=num_songs,
                                      mode='train')
    # train NN
    if num_songs == 1:
        train([trained_song['pitches'][:num_melody]], pitch_net, mode='pitch')
        train([trained_song['durations'][:num_melody]], duration_net, mode='duration')
    else:
        train(songs_pitches, pitch_net, mode='pitch')
        train(songs_durations, duration_net, mode='duration')


if __name__ == '__main__':
    tf.app.run()
