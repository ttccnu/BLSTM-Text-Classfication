import numpy as np
import tensorflow as tf
import random
import time


class RNN(object):
    def __init__(self, sess=None):
        self.embedding_size = 30
        self.session = sess
        pass

    def blstm(self, train_data, train_label, test_data, test_label):
        X = tf.placeholder(tf.float32, [1, None, 100])
        Y = tf.placeholder(tf.float32, [10])
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size, forget_bias=1.0)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                   lstm_bw_cell, X,
                                                                   dtype=tf.float32)

        fw_output, bw_output = outputs[0], outputs[1]  # 前向、后向输出，每个输出都是【1，n, embeding_size】,其中n为这个序列的长度（有多少个词）
        fw_output = fw_output[:, -1, :]  # 取最后一个输出
        bw_output = bw_output[:, -1, :]
        concated = tf.concat([fw_output, bw_output], 1)  # 将其拼接为一个(1,20)的向量

        # fc layer
        W1 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1, seed=2))
        W2 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1, seed=2))
        bias1 = tf.Variable(tf.constant(0.1, shape=[30]))
        bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
        h = tf.nn.relu(tf.matmul(concated, W1) + bias1)
        y_pre = tf.nn.softmax(tf.matmul(h, W2) + bias2)
        cross_entropy = -tf.reduce_sum(Y * tf.log(y_pre))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_pre, -1), tf.argmax(Y, -1))
        accuracy = tf.cast(correct_prediction, tf.float32)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        record = []
        for epoch in range(1000):   # Total epoch
            rand_id = np.arange(len(train_label))
            random.shuffle(rand_id)
            acc = 0
            st = time.time()
            for n in range(200):         # 随机下降的batch个数
                sess.run(train_step, feed_dict={X: train_data[rand_id[n]], Y: train_label[rand_id[n]]})
            for t in range(1000):       # 测试的个数
                a = sess.run(accuracy, feed_dict={X: test_data[t], Y: test_label[t]})
                acc += a
            record.append(float(acc/10))
            et = time.time()
            print("epoch= ", epoch, "   accuracy= ", float(acc/10), "%", "    time= ", et-st)
            pass
        pass
    pass
