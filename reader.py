#!/usr/bin/env python
# coding=utf8

"""
数据处理类: 词表生成、单词查找等功能
"""

from __future__ import print_function

import collections
import cPickle

import numpy as np

###############################################################################
# 数据处理: 过滤低频词条
class Reader(object):
    """
    数据处理类: 词表生成、单词查找等功能
    """

    def __init__(self, min_count, context_window, flag_padding):
        self.min_count = min_count 
        self.context_window = context_window
        self.flag_padding = flag_padding

        # 设置UNK, Li, Ri几个默认词向量
        self.collections_counter = collections.Counter()
        self.word_counter = [("UNK", 0)]
        if self.flag_padding:
            for i in range(self.context_window - 1, -1, -1):
                self.word_counter.append(("L" + str(i), 0))
            for i in range(self.context_window):
                self.word_counter.append(("R" + str(i), 0))

        self.vocab = dict()
        self.reverse_vocab = dict()
        self.vocab_size = 0

    def update(self, word_list):
        """ 输入单词列表，更新频次计数 """
        self.collections_counter.update(word_list)
        return 0

    def gen_vocab(self):
        """ 根据频次信息生成词表 """
        num = len(self.collections_counter)
        self.word_counter.extend(self.collections_counter.most_common())
        self.word_counter = filter(lambda x: x[1] >= self.min_count, self.word_counter)
        self.vocab_size = len(self.word_counter)
        for word, _ in self.word_counter:
            self.vocab[word] = len(self.vocab)
        # 生成逆向词典
        self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
        # dump词典明文
        with open("vocab.txt", "w") as fp:
            for key in self.vocab:
                fp.write("%s\t%d\n" % (key, self.vocab[key]))

        return self.vocab_size

    def save(self, vocab_path):
        """ 将词典信息序列化 """
        with open(vocab_path, "wb") as fp:
            cPickle.dump(self.vocab, fp, -1)
            cPickle.dump(self.reverse_vocab, fp, -1)
            cPickle.dump(self.word_counter, fp, -1)
        return 0

    def load(self, vocab_path):
        """ 加载词典 """
        with open(vocab_path, "rb") as fp:
            self.vocab = cPickle.load(fp)
            self.reverse_vocab = cPickle.load(fp)
            self.word_counter = cPickle.load(fp)
        return 0

    def lookup(self, word):
        """ 单词查询不到默认返回unk """
        ind = self.vocab.get(word, 0)
        return ind

    def reverse_lookup(self, ind):
        """ 反向查询 """
        return self.reverse_vocab[ind]

    def gen_cbow_data(self, line):
        """ 输入每行句子，输出tf训练数据格式 """
        x = list()
        y = list()
        words = list()
        # 如果设置了padding则在头尾插入padding标记
        if self.flag_padding:
            for i in range(self.context_window - 1, -1, -1):
                words.append("L" + str(i))
        words.extend(line.strip().split())
        if self.flag_padding:
            for i in range(self.context_window):
                words.append("R" + str(i))
        ids = map(lambda x: self.lookup(x), words)

        # 解析出窗口向量和目标向量
        begin = self.context_window
        end = len(words) - self.context_window
        for i in range(begin, end):
            x.extend(ids[i - self.context_window: i])
            x.extend(ids[i + 1: i + self.context_window + 1])
            y.extend(ids[i: i + 1])

        # 转换成tf适配格式输入到feed_dict
        x_input = np.array(x).reshape([-1, self.context_window * 2])
        y_input = np.array(y).reshape([-1, 1])
        return x_input, y_input

    def gen_skipgram_data(self, line):
        """ 输入每行句子，输出tf训练数据格式 """
        x = list()
        y = list()
        words = list()
        # 如果设置了padding则在头尾插入padding标记
        if self.flag_padding:
            for i in range(self.context_window - 1, -1, -1):
                words.append("L" + str(i))
        words.extend(line.strip().split())
        if self.flag_padding:
            for i in range(self.context_window):
                words.append("R" + str(i))
        ids = map(lambda x: self.lookup(x), words)

        # 解析出窗口向量和目标向量
        begin = self.context_window
        end = len(words) - self.context_window
        for i in range(begin, end):
            for j in range(self.context_window):
                x.extend(ids[i: i + 1])
                x.extend(ids[i: i + 1])
            y.extend(ids[i - self.context_window: i])
            y.extend(ids[i + 1: i + self.context_window + 1])

        # 转换成tf适配格式输入到feed_dict
        x_input = np.array(x)
        y_input = np.array(y).reshape([-1, 1])
        return x_input, y_input

