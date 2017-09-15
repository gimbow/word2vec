#!/usr/bin/env python
# coding=utf8

"""
word2vec: 窗口concat + negative sample实现
"""

from __future__ import print_function

import math
import random
import collections
import cPickle

import numpy as np
import tensorflow as tf
import reader as rd

###############################################################################
# 解析命令行参数
flags = tf.flags
flags.DEFINE_string("data_path", "data.txt", "训练数据")
flags.DEFINE_string("vocab_path", "vocab.pkl", "词典文件")
flags.DEFINE_integer("step_size", 2000, "间隔step打印调试")
flags.DEFINE_integer("min_count", 0, "低频词条阈值")
flags.DEFINE_integer("emb_size", 128, "词向量维度")
flags.DEFINE_integer("epoch_size", 20, "迭代次数")
flags.DEFINE_bool("flag_padding", False, "是否设置padding")
flags.DEFINE_integer("context_window", 1, "单边窗口大小")
flags.DEFINE_integer("num_sampled", 64, "负采样个数")
flags.DEFINE_float("learning_rate", 0.1, "学习率")
flags.DEFINE_integer("valid_window", 100, "验证集在词表中范围")
flags.DEFINE_integer("valid_size", 16, "验证集大小")
flags.DEFINE_string("model_path", "concat.pkl", "模型文件")
flags.DEFINE_string("log_path", "logs", "log记录位置")
FLAGS = flags.FLAGS

###############################################################################
class ConcatModel(object):
    """ concat + negative sample 模型 """

    # 构建计算图
    def __init__(self, lr, vocab_size, emb_size, context_window, num_sampled, 
            model_path, valid_examples):
        self.lr = lr
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.context_size = 2 * context_window
        self.num_sampled = num_sampled
        self.model_path = model_path

        # 输入数据
        self.xs = tf.placeholder(tf.int32, shape=[None, self.context_size], name="xs")
        self.ys = tf.placeholder(tf.int32, shape=[None, 1], name="ys")

        # 定义模型参数
        embeddings = tf.Variable(tf.random_uniform([vocab_size, emb_size], -1.0, 1.0), name="emb")
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, emb_size], 
            stddev=1.0 / math.sqrt(emb_size)), name="weight")
        nce_biases = tf.Variable(tf.zeros([vocab_size]), name="bias")
        tf.summary.histogram("nce_weights", nce_weights)
        tf.summary.histogram("nce_biases", nce_biases)
        # 查询窗口向量
        with tf.device("/cpu:0"):
            self.embed = tf.nn.embedding_lookup(embeddings, self.xs)
        # 平均
        #self.context_emb = tf.reduce_mean(self.embed, 1)
        # 拼接
        self.concat_emb = tf.reshape(self.embed, [-1, self.context_size * self.emb_size]) 
        # 窗口向量经过layer，输出和目标向量相同维度
        # dense.kenerl, dense.bias可以取出层对应的参数
        # denselayer返回是一个tensor, 维度由units定义
        self.context_emb = tf.layers.dense(
            inputs=self.concat_emb,
            units=emb_size,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1')

        # loss
        self.nce_loss = tf.nn.nce_loss(weights=nce_weights, 
            biases=nce_biases, 
            labels=self.ys,
            inputs=self.context_emb, 
            num_sampled=num_sampled, 
            num_classes=vocab_size)
        self.loss = tf.reduce_mean(self.nce_loss)
        tf.summary.scalar("loss", self.loss)

        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # 计算cosine距离
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm
        with tf.device("/cpu:0"):
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

        # 保存和加载模型
        self.saver = tf.train.Saver()

    def save(self, sess):
        """ 序列化模型 """
        self.saver.save(sess, self.model_path)
        return 0

    def load(self, sess):
        """ 反序列化模型 """
        self.saver.restore(sess, self.model_path)
        return 0


###############################################################################
def main(_):
    """ 设置训练的总流程 """
    # 解析生成vocab信息
    reader = rd.Reader(FLAGS.min_count, FLAGS.context_window, FLAGS.flag_padding)
    fp = open(FLAGS.data_path, "r")
    for line in fp:
        words = line.strip().split()
        reader.update(words)
    fp.close()
    vocab_size = reader.gen_vocab()
    reader.save(FLAGS.vocab_path)

    # 构建concat模型计算图
    valid_examples = np.random.choice(FLAGS.valid_window, FLAGS.valid_size, replace=False)
    model = ConcatModel(FLAGS.learning_rate, vocab_size, FLAGS.emb_size, 
        FLAGS.context_window, FLAGS.num_sampled, FLAGS.model_path, valid_examples)

    # session 和 初始化
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 读取每行句子进行训练
    step = 0
    average_loss = 0.0
    fp = open(FLAGS.data_path, "r")
    for epoch in range(FLAGS.epoch_size + 1):
        for line in fp:
            step += 1
            # 输入训练数据 并 更新网络
            xs, ys = reader.gen_cbow_data(line)
            feed_dict = {model.xs: xs, model.ys: ys}
            _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            average_loss += loss

            # 打印loss
            if step % FLAGS.step_size == 0:
                average_loss /= FLAGS.step_size
                print("epoch ", epoch, ", step ", step, ", average_loss = ", average_loss)
                average_loss = 0.0
                # tensorboard打点
                train_result = sess.run(merged, feed_dict=feed_dict)
                writer.add_summary(train_result, step)

            # 调试验证集的邻近词
            if step % (FLAGS.step_size * 10) == 0:
                sim = model.similarity.eval(session=sess)
                for i in xrange(FLAGS.valid_size):
                    valid_word = reader.reverse_lookup(valid_examples[i])
                    top_k = 8
                    nearest = (-sim[i,:]).argsort()[1: top_k + 1]
                    log_str = "nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reader.reverse_lookup(nearest[k])
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)

        # 将文件指针移动到开始
        fp.seek(0) 

    # 保存训练后的模型
    model.save(sess)
    fp.close()
    sess.close() 

    return 0

###############################################################################
if __name__ == "__main__":
    tf.app.run()

