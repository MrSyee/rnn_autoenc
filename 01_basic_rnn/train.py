
# basic RNN using one-hot encdoing time series song data

import tensorflow as tf
import numpy as np
from datetime import datetime

import RNN_model as rnn

now = datetime.now()

def main(_):
    melody = [50, 10, 17, 17, 17, 15, 13,  17, 17, 17, 17, 17, 15, 13,  17, 50, 17, 17, 15, 13,  17, 17, 17, 17, 15, 17, 15, 13,   10, 10, 17, 17, 17, 17, 15, 13,  17, 50, 17, 15, 17, 15, 13,  20, 50, 17, 17, 17, 15, 13,  17, 15, 15, 15, 15, 17, 15, 13,   10, 22, 22, 50, 10, 10, 13,  10, 22, 22, 50, 22, 22, 22,  20, 50, 20, 20, 20, 25, 24,  20, 50, 17, 15, 13,   10, 22, 22, 50, 10, 10, 13,  10, 22, 22, 50, 22, 22, 22,  20, 50, 20, 20, 20, 25, 24,  25, 24, 50, 17, 20,   22, 22, 17, 20, 17, 20,  22, 22, 20, 25, 25, 24, 25,  20, 50, 20, 25, 25, 24, 25,  29, 27, 25, 24, 20,   22, 22, 17, 20, 17, 20,  22, 22, 20, 25, 25, 24, 25,  20, 50, 20, 25, 25, 24, 25,  29, 27, 25, 24, 24]
    rhythm = [2, 2, 2, 2, 4, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 4, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2,   2, 2, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2,   4, 3, 1, 2, 2, 2, 2,  4, 3, 1, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  8, 2, 2, 2, 2,   4, 3, 1, 2, 2, 2, 2,  4, 3, 1, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  2, 6, 4, 2, 2,  4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 4, 4, 2, 2,   4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,  4, 2, 2, 2, 2, 2, 2,  4, 4, 4, 2, 2,]
    # set sample song data (Roly Poly)
    train_melody = []
    train_melody.append(melody)
    train_melody.append(rhythm)

    song_length = len(melody)

    # Train the RNN model
    step = 1000

    model = rnn.RNN(song_length=song_length,
                    mode='train')
    '''
    x_data, y_data = model.data2idx(train_melody)
    x_data = np.reshape(x_data, [model.batch_size, len(x_data)])
    y_data = np.reshape(y_data, [model.batch_size, len(y_data)])
    '''

    model.build()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + now.strftime("%Y%m%d-%H%M%S")+ "/", sess.graph)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            x_data, y_data = model.data2idx(train_melody)
            x_data = np.reshape(x_data, [model.batch_size, x_data.shape[1]])
            y_data = np.reshape(y_data, [model.batch_size, y_data.shape[1]])

            loss_val, _ = sess.run([model.loss, model.train], feed_dict={model.X: x_data, model.Y: y_data})
            result = sess.run(model.prediction, feed_dict={model.X: x_data})

            summary = sess.run(summary_op, feed_dict={model.X: x_data, model.Y: y_data})

            #result_str = [model.idx2char[c] for c in np.squeeze(result)]

            print("{:4d}  loss: {:.5f}".format(i, loss_val))

            writer.add_summary(summary, i)

        saver.save(sess, "./save/" + now.strftime("%Y%m%d-%H%M%S") + "/rnn_model.ckpt")

if __name__ == '__main__':
    tf.app.run()
