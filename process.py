#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :NLP 
@File    :process.py
@Author  :JackHCC
@Date    :2022/6/12 14:11 
@Desc    :process raw data and cut sentences

'''

import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg

ROOT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
JIEBA_USER_DICT = os.path.join(CONFIG_PATH, 'jieba_user_dict.txt')
STOP_WORDS = os.path.join(CONFIG_PATH, 'stop_words.txt')
POS_DICT = os.path.join(CONFIG_PATH, 'POS_dict.txt')

# 加载词性筛选配置
FLAG_LIST = []
with open(POS_DICT, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        FLAG_LIST.append(line)
print(FLAG_LIST)


def chinese_word_cut(article):
    jieba.load_userdict(JIEBA_USER_DICT)
    jieba.initialize()
    try:
        stop_word_list = open(STOP_WORDS, encoding='utf-8')
    except:
        stop_word_list = []
        print("Error in stop_words file")
    stop_list = []

    for line in stop_word_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    word_list = []
    # jieba分词
    seg_list = psg.cut(article)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in FLAG_LIST:
            word_list.append(word)

    return ' '.join(word_list)


def data_word_cut(data_path=os.path.join(DATA_PATH, 'data.xlsx')):
    data = pd.read_excel(data_path)
    data["content_cut"] = data.content.apply(chinese_word_cut)
    print("Data process get columns info: ", data.columns)
    return data


if __name__ == "__main__":
    data_word_cut()

