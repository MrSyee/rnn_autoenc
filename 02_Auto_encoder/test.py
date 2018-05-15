
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np

import RNN_AE_model_decoder_feedback as rnn_AE
import utils

sv_datetime = "20180515-213956"

util = utils.Util()

def test(trained_data, len_data, mode):
    # Test the RNN model
    char2idx = util.getchar2idx(mode=mode)

    enc_model = rnn_AE.LSTMAutoEnc(song_length=len_data,
                               batch_size=1,
                               mode='test')
    dec_model = rnn_AE.LSTMAutoEnc(song_length=len_data,
                               batch_size=1,
                               mode='test')

    enc_out_state = enc_model.encoder(scopename=mode)
    dec_model.decoder(scopename=mode)
    #dec_model.build()

    # decoder input
    test_data = util.data2idx(trained_data, char2idx)

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
        #dec_in_state = np.zeros([dec_model.batch_size, dec_model.enc_cell.state_size])

        prediction = sess.run(dec_model.prediction, feed_dict={dec_model.Dec_input: test_data,
                                                           dec_model.Dec_state: dec_in_state})

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
        result[result == 'Rest'] = 0
        result = list(result)
        trained_data = np.array(trained_data)
        trained_data[trained_data == 'Rest'] = 0
        trained_data = list(trained_data)
        error = [int(x) - int(y) for x, y in zip(result, trained_data)]
        print("error : ", error)
    else:
        error = [x - y for x, y in zip(result, trained_data)]
        print("error : ", error)

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

    pitches = test([trained_song['pitches']], trained_song['length'], mode='pitch')
    durations = test([trained_song['durations']], trained_song['length'], mode='duration')
    # make midi file
    util.song2midi(pitches, durations)

if __name__ == '__main__':
    tf.app.run()


