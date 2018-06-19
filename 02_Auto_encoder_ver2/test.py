
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np

from RNN_AE_model_decoder_feedback_melody import LSTMAutoEnc
import utils

sv_datetime = "20180606-174059"

util = utils.Util()

def test(song, len_data):
    # Test the RNN model
    # make char2idx
    p_char2idx = util.getchar2idx(mode="pitch")
    d_char2idx = util.getchar2idx(mode="duration")

    enc_model = LSTMAutoEnc(song_length=len_data,
                            batch_size=1,
                            mode='test')
    dec_model = LSTMAutoEnc(song_length=len_data,
                            batch_size=1,
                            mode='test')

    p_enc_state, d_enc_state = enc_model.encoder()
    dec_model.decoder()

    song_pitch = [song['pitches']]
    song_duration = [song['durations']]

    # encoder input
    pitch_data = util.data2idx(song_pitch, p_char2idx)
    pitch_data = np.reshape(pitch_data, [enc_model.batch_size, pitch_data.shape[1]])
    duration_data = util.data2idx(song_duration, d_char2idx)
    duration_data = np.reshape(duration_data, [enc_model.batch_size, duration_data.shape[1]])

    p_enc_saver = tf.train.Saver(var_list=enc_model.p_enc_vars)
    d_enc_saver = tf.train.Saver(var_list=enc_model.d_enc_vars)
    p_dec_saver = tf.train.Saver(var_list=dec_model.p_dec_vars)
    d_dec_saver = tf.train.Saver(var_list=dec_model.d_dec_vars)

    with tf.Session() as sess:
        p_enc_saver.restore(sess, "./save/" + sv_datetime + "/pitch_enc_model.ckpt")
        d_enc_saver.restore(sess, "./save/" + sv_datetime + "/duration_enc_model.ckpt")
        p_dec_saver.restore(sess, "./save/" + sv_datetime + "/pitch_dec_model.ckpt")
        d_dec_saver.restore(sess, "./save/" + sv_datetime + "/duration_dec_model.ckpt")
        '''
        for i in model.enc_vars:
            print (i)
        for i in model.dec_vars:
            print (i)
        '''

        p_state, d_state  = sess.run([p_enc_state, d_enc_state], feed_dict={enc_model.p_input: pitch_data,
                                                                            enc_model.d_input: duration_data})

        # avarage encoder state output
        p_d_s = np.mean(p_state, 0).reshape([dec_model.batch_size, dec_model.p_enc_cell.state_size])
        d_d_s = np.mean(d_state, 0).reshape([dec_model.batch_size, dec_model.d_enc_cell.state_size])

        prediction = [dec_model.p_prediction, dec_model.d_prediction]

        p, d = sess.run(prediction, feed_dict={dec_model.p_Dec_state: p_d_s,
                                               dec_model.d_Dec_state: d_d_s})

        p_result = util.idx2char(p, 'pitch')
        d_result = util.idx2char(d, 'duration')

        # print : result - trained_data
        print_error(p_result, song_pitch, 'pitch')
        print_error(d_result, song_duration, 'duration')

        return p_result, d_result

def print_error(result, trained_data, mode):
    # print : result - trained_data
    trained_data = trained_data[0]
    print("trained_data : ", trained_data)
    if mode == 'pitch':
        result = np.array(result)
        result[result == 'Rest'] = 0
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

    pitches, durations = test(trained_song, trained_song['length'])
    # make midi file
    util.song2midi(pitches, durations, '/generate', sv_datetime)

if __name__ == '__main__':
    tf.app.run()


