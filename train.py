# -*- coding: utf-8 -*- 
# @Time : 2018/8/2 9:22 
# @Author : Allen 
# @Site :  训练验证码

import numpy as np
import tensorflow as tf
from captchaIndentify import Captcha
from utils import is_dir, get_dir
import os

class CCNN():
    def __init__(self):
        self.captcha = Captcha()
        self.text, image = self.captcha.get_captcha_text_and_image()
        self.IMAGE_HEIGHT = 60
        self.IMAGE_WIDTH = 160
        self.MAX_CAPTCHA = len(self.text)
        self.CHAR_SET_LEN = self.captcha.get_len_char_set()
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None, self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)
        self.model_path = is_dir(get_dir()+"ckpt"+os.sep)
        self.output = self.creat_captcha_cnn()

    def get_weight(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def get_bias(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.1))

    def creat_captcha_cnn(self):
        x = tf.reshape(self.X, shape=[-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1])
        w_c1 = self.get_weight([3, 3, 1, 32])
        b_c1 = self.get_bias([32])
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        w_c2 = self.get_weight([3, 3, 32, 64])
        b_c2 = self.get_bias([64])
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        w_c3 = self.get_weight([3, 3, 64, 64])
        b_c3 = self.get_bias([64])
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        w = conv3.get_shape().as_list()
        w_d = self.get_weight([w[1] * w[2] * w[3], 1024])
        b_d = self.get_bias([1024])
        dense = tf.reshape(conv3, [-1, w[1] * w[2] * w[3]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        w_out = self.get_weight([1024, self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        b_out = self.get_bias([self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        out = tf.add(tf.matmul(dense, w_out), b_out)

        return out

    def convert2gray(self, img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            # 上面的转法较快，正规转法如下
            # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.MAX_CAPTCHA:
            raise ValueError('验证码最长4个字符')

        vector = np.zeros(self.MAX_CAPTCHA * self.CHAR_SET_LEN)

        def char2pos(c):
            if c == '_':
                k = 62
                return k
            k = ord(c) - 48
            if k > 9:
                k = ord(c) - 55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k

        for i, c in enumerate(text):
            # print text
            idx = i * self.CHAR_SET_LEN + char2pos(c)
            # print i,CHAR_SET_LEN,char2pos(c),idx
            vector[idx] = 1
        return vector

    # 向量转回文本
    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % self.CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    def get_next_batch(self, batch_size=128):
        batch_x = np.zeros([batch_size, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, self.MAX_CAPTCHA * self.CHAR_SET_LEN])

        def wrap_gen_captcha_text_and_image():
            while True:
                text, image = self.captcha.get_captcha_text_and_image()
                if image.shape == (60, 160, 3):
                    return text, image

        for i in range(batch_size):
            text, image = wrap_gen_captcha_text_and_image()
            image = self.convert2gray(image)

            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(text)
            return batch_x, batch_y

    def train(self):
        import time
        start_time = time.time()
        # out = self.creat_captcha_cnn()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
        predict = tf.reshape(self.output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_inx_l = tf.argmax(tf.reshape(self.Y, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_inx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y = self.get_next_batch(64)
                _, loss_ = sess.run([optimizer, loss],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), step, loss_)

                if step % 100 == 0 and step != 0:
                    batch_x_test, batch_y_test = self.get_next_batch()
                    acc = sess.run(accuracy,
                                   feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 0.75})
                    print("第%s轮，经度为：%s" % (str(step), str(acc)))

                    if acc > 0.9:
                        saver.save(sess, self.model_path+"capcha.model", global_step=step)
                        break
                step += 1

    def crack_captcha(self, captcha_image):

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path+'capcha.model')
            predict = tf.argmax(tf.reshape(self.output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [captcha_image], self.keep_prob: 1})
            text = text_list[0].tolist()
            return text


if __name__ == '__main__':
    ccnn = CCNN()
    train = 0
    if train == 0:
        ccnn.train()
    else:
        captcha = Captcha()
        for i in range(10):
            text, image = captcha.get_captcha_text_and_image()
            image = ccnn.convert2gray(image).flatten() / 255
            predict_text = ccnn.vec2text(ccnn.crack_captcha(image))
            print("正确验证码：{} 预测验证码{}".format(text, predict_text))
