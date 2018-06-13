
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np

import RNN_AE_model_decoder_feedback as rnn_AE
import utils

sv_datetime = "20180606-221614_6songs"

util = utils.Util()

def test(trained_data, len_data, mode):
    # Test the RNN model
    char2idx = util.getchar2idx(mode=mode)

    enc_model = rnn_AE.LSTMAutoEnc(song_length=len_data,
                                   batch_size=len(trained_data),
                                   mode='test')
    dec_model = rnn_AE.LSTMAutoEnc(song_length=len_data,
                                   batch_size=1,
                                   mode='test')

    enc_out_state = enc_model.encoder(scopename=mode)
    dec_model.decoder(scopename=mode)

    # encoder input
    state_sample_data = util.data2idx(trained_data, char2idx)

    enc_saver = tf.train.Saver(var_list=enc_model.enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_model.dec_vars)
    with tf.Session() as sess:
        enc_saver.restore(sess, "./save/" + sv_datetime + "/enc_{}_model.ckpt".format(mode))
        dec_saver.restore(sess, "./save/" + sv_datetime + "/dec_{}_model.ckpt".format(mode))
        '''
        for i in model.enc_vars:
            print (i)
        for i in model.dec_vars:
            print (i)
        '''

        enc_out_state = sess.run(enc_out_state, feed_dict={enc_model.Enc_input: state_sample_data})

        # avarage encoder state output
        dec_in_state = np.mean(enc_out_state, 0).reshape([dec_model.batch_size, dec_model.enc_cell.state_size])

        prediction = sess.run(dec_model.prediction, feed_dict={dec_model.Dec_state: dec_in_state})

        result = util.idx2char(prediction, mode)
        print("result : ", result)

        # print : result - trained_data
        print_error(result, trained_data, mode)

        return result

def print_error(result, trained_data, mode):
    # print : result - trained_data
    trained_data = trained_data[0]
    print("trained_data : ", trained_data)
    if mode == 'pitch':
        result = np.array(result)
        result['Rest' == result] = 0
        result = list(result)
        trained_data = np.array(trained_data)
        trained_data[trained_data == 'Rest'] = 0
        trained_data = list(trained_data)
        error = [abs(int(x) - int(y)) for x, y in zip(result, trained_data)]
        print("error : ", error)
        print("total error : ", sum(error))
    else:
        error = [abs(x - y) for x, y in zip(result, trained_data)]
        print("error : ", error)
        print("total error : ", sum(error))

def main(_):
    '''
    # load one midi file
    filename = 'test.mid'
    trained_song = util.get_one_song(filename)
    print("One song load : {}".format(filename))
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

    # output song
    pitches = test(songs_pitches, min(songs_len), mode='pitch')
    durations = test(songs_durations, min(songs_len), mode='duration')

    # make midi file
    util.song2midi(pitches, durations, sv_datetime)

if __name__ == '__main__':
    tf.app.run()

