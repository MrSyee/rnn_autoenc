
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
from datetime import datetime

import RNN_AE_model_decoder_feedback as rnn_AE

now = datetime.now()

# roly poly : [50, 10, 17, 17, 17, 15, 13,  17, 17, 17, 17, 17, 15, 13,  17, 50, 17, 17, 15, 13,  17, 17, 17, 17, 15, 17, 15, 13,   10, 10, 17, 17, 17, 17, 15, 13,  17, 50, 17, 15, 17, 15, 13,  20, 50, 17, 17, 17, 15, 13,  17, 15, 15, 15, 15, 17, 15, 13,   10, 22, 22, 50, 10, 10, 13,  10, 22, 22, 50, 22, 22, 22,  20, 50, 20, 20, 20, 25, 24,  20, 50, 17, 15, 13,   10, 22, 22, 50, 10, 10, 13,  10, 22, 22, 50, 22, 22, 22,  20, 50, 20, 20, 20, 25, 24,  25, 24, 50, 17, 20,   22, 22, 17, 20, 17, 20,  22, 22, 20, 25, 25, 24, 25,  20, 50, 20, 25, 25, 24, 25,  29, 27, 25, 24, 20,   22, 22, 17, 20, 17, 20,  22, 22, 20, 25, 25, 24, 25,  20, 50, 20, 25, 25, 24, 25,  29, 27, 25, 24, 24]
#             [2, 2, 2, 2, 4, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 4, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2,   2, 2, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2,   4, 3, 1, 2, 2, 2, 2,  4, 3, 1, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  8, 2, 2, 2, 2,   4, 3, 1, 2, 2, 2, 2,  4, 3, 1, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  2, 6, 4, 2, 2,  4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 4, 4, 2, 2,   4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 4, 4, 2, 2,]
# River flows in you : [22, 21, 22, 21, 22, 17, 22, 15, 15, 15, 10, 14, 22, 21, 22, 21, 22, 17, 22, 15, 15, 15, 10, 14, 22, 21, 22, 22, 10, 21, 22, 22, 10, 17, 22, 22, 10, 15, 10, 14, 15, 17, 14, 12, 10, 9, 10, 10, 5, 10, 12, 14, 14, 15, 17, 15, 14, 12, 22, 21, 22, 22, 10, 21, 22, 22, 10, 17, 22, 22, 10, 15, 10, 14, 15, 17, 26, 24, 17, 24, 22, 21, 22, 10, 12, 14, 5, 10, 14, 15, 17, 5, 14, 15, 13, 12, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 24, 26, 27, 29, 26, 24, 22, 21, 12, 22, 26, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 24, 26, 27, 29, 26, 24, 22, 21, 12, 22, 22, 21, 22]

def main(_):
    melody = [
             [22, 21, 22, 21, 22, 17, 22, 15, 15, 15, 10, 14, 22, 21, 22, 21, 22, 17, 22, 15, 15, 15, 10, 14, 22, 21, 22, 22, 10, 21, 22, 22, 10, 17, 22, 22, 10, 15, 10, 14, 15, 17, 14, 12, 10, 9, 10, 10, 5, 10, 12, 14, 14, 15, 17, 15, 14, 12, 22, 21, 22, 22, 10, 21, 22, 22, 10, 17, 22, 22, 10, 15, 10, 14, 15, 17, 26, 24, 17, 24, 22, 21, 22, 10, 12, 14, 5, 10, 14, 15, 17, 5, 14, 15, 13, 12, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 24, 26, 27, 29, 26, 24, 22, 21, 12, 22, 26, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 10, 17, 10, 22, 24, 22, 21, 22, 24, 26, 27, 29, 26, 24, 22, 21, 12, 22, 22, 21, 22], \
            [50, 10, 17, 17, 17, 15, 13, 17, 17, 17, 17, 17, 15, 13, 17, 50, 17, 17, 15, 13, 17, 17, 17, 17, 15, 17, 15, 13,
             10, 10, 17, 17, 17, 17, 15, 13, 17, 50, 17, 15, 17, 15, 13, 20, 50, 17, 17, 17, 15, 13, 17, 15, 15, 15, 15, 17,
             15, 13, 10, 22, 22, 50, 10, 10, 13, 10, 22, 22, 50, 22, 22, 22, 20, 50, 20, 20, 20, 25, 24, 20, 50, 17, 15, 13,
             10, 22, 22, 50, 10, 10, 13, 10, 22, 22, 50, 22, 22, 22, 20, 50, 20, 20, 20, 25, 24, 25, 24, 50, 17, 20, 22, 22,
             17, 20, 17, 20, 22, 22, 20, 25, 25, 24, 25, 20, 50, 20, 25, 25, 24, 25, 29, 27, 25, 24, 20, 22, 22, 17, 20, 17,
             20, 22, 22, 20, 25, 25, 24, 25, 20, 50, 20, 25, 25, 24, 25, 29, 27, 25, 24, 24]
             ]

    # set sample song data (Roly Poly)
    train_melody = melody
    song_length = np.shape(train_melody)[1]

    # Train the RNN model
    step = 1000

    model = rnn_AE.LSTMAutoEnc(song_length=song_length,
                               batch_size=len(train_melody),
                               mode='train')
    '''
    x_data, y_data = model.data2idx(train_melody)
    x_data = np.reshape(x_data, [model.batch_size, len(x_data)])
    y_data = np.reshape(y_data, [model.batch_size, len(y_data)])
    '''

    model.build()

    enc_saver = tf.train.Saver(var_list=model.enc_vars)
    dec_saver = tf.train.Saver(var_list=model.dec_vars)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + now.strftime("%Y%m%d-%H%M%S")+ "/", sess.graph)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            x_data = model.data2idx(train_melody)
            x_data = np.reshape(x_data, [model.batch_size, x_data.shape[1]])

            loss_val, _ = sess.run([model.loss, model.train], feed_dict={model.Enc_input: x_data,
                                                                         model.Dec_input: x_data})
            result = sess.run(model.prediction, feed_dict={model.Enc_input: x_data,
                                                           model.Dec_input: x_data})

            summary = sess.run(summary_op, feed_dict={model.Enc_input: x_data,
                                                      model.Dec_input: x_data})

            #result_str = [model.idx2char[c] for c in np.squeeze(result)]

            print("{:4d}  loss: {:.5f}".format(i, loss_val))

            writer.add_summary(summary, i)

        enc_saver.save(sess, "./save/" + now.strftime("%Y%m%d-%H%M%S") + "/enc_model.ckpt")
        dec_saver.save(sess, "./save/" + now.strftime("%Y%m%d-%H%M%S") + "/dec_model.ckpt")

if __name__ == '__main__':
    tf.app.run()
