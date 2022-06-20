#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Chinese-Keyphrase-Extraction 
@File    :distribution_statistics.py
@Author  :JackHCC
@Date    :2022/6/17 11:08 
@Desc    :Subject distribution of statistical words

'''
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from utils import save_pickle
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置标题标注和字体大小
plt.rcParams.update({"font.size": 14})  # 此处必须添加此句代码方可改变标题字体大小
# 添加网格显示
plt.grid(linestyle="-", alpha=0.5, linewidth=1.5)


def get_topic_and_keyword(path):
    data = pd.read_excel(path)
    target_data = data[["type", "key_phrase"]]
    topic_keyword_list = []
    for idx, row in target_data.iterrows():
        topic = row["type"].strip()
        key_words = row["key_phrase"].strip().split(";")
        topic_keyword_list.append((topic, key_words))
    return topic_keyword_list


def topic_statistic(topic_keyword_list):
    topic_set = set()
    keyword_vocab = {}
    topic_keyword_count = defaultdict(dict)
    for topic, keywords in topic_keyword_list:
        topic_set.add(topic)
        for keyword in keywords:
            keyword_vocab[keyword] = keyword_vocab.get(keyword, 0) + 1
            topic_keyword_count[topic][keyword] = topic_keyword_count[topic].get(keyword, 0) + 1

    for word in list(topic_keyword_count.keys()):
        for topic, topic_dict in topic_keyword_count.items():
            if word not in list(topic_keyword_count[topic].keys()):
                topic_keyword_count[topic][word] = 0

    topic_set = {topic: idx for idx, topic in enumerate(topic_set)}

    return topic_set, keyword_vocab, topic_keyword_count


def get_topic_topk_words(topic_keyword_count, k=10):
    top_k_dict = {}
    for topic, keywords_dict in topic_keyword_count.items():
        keywords_dict_counter = Counter(keywords_dict)
        topk_words = keywords_dict_counter.most_common(k)
        top_k_dict[topic] = topk_words

    return top_k_dict


def show_topic_topk_words(top_k_dict):
    for topic, word_count in top_k_dict.items():
        words = [word for word, count in word_count]
        words_str = ";".join(words)
        print("主题：", topic, " Top-" + str(len(words)) + "关键词：", words_str)


def keyword_statistic(topic_keyword_list):
    topic_set = set()
    keyword_vocab = {}
    keyword_topic_count = defaultdict(dict)
    for topic, keywords in topic_keyword_list:
        topic_set.add(topic)
        for keyword in keywords:
            keyword_vocab[keyword] = keyword_vocab.get(keyword, 0) + 1
            keyword_topic_count[keyword][topic] = keyword_topic_count[keyword].get(topic, 0) + 1

    for topic in topic_set:
        for keyword, topic_dict in keyword_topic_count.items():
            if topic not in list(keyword_topic_count[keyword].keys()):
                keyword_topic_count[keyword][topic] = 0

    topic_set = {topic: idx for idx, topic in enumerate(topic_set)}

    return topic_set, keyword_vocab, keyword_topic_count


def get_topic_distribution(keyword_topic_count, topic_set):
    word_num = len(keyword_topic_count)
    topic_num = len(topic_set)
    topic_distribution = np.zeros((word_num, topic_num))
    idx = 0
    keyword_dict = {}
    for keyword, topic_dict in keyword_topic_count.items():
        keyword_dict[keyword] = idx
        for topic, count in topic_dict.items():
            topic_distribution[idx][topic_set[topic]] += count
        idx += 1

    topic_distribution_final = topic_distribution / topic_distribution.sum(axis=1)[:, np.newaxis]
    print("distribution shape:", topic_distribution_final.shape)

    return topic_distribution_final, keyword_dict


def draw_topic_distribution(word, topic_distribution, keyword_dict, topic_set):
    """
    画指定word的主题分布图
    @param word: 想要观察主题分布的词语
    @param topic_distribution: 主题分布矩阵
    @param keyword_dict: 词汇检索表
    @param topic_set: 主题检索表
    """
    assert word in list(keyword_dict.keys())
    topic = list(topic_set.keys())

    # 设置坐标标签标注和字体大小
    plt.xlabel("主题", fontsize=12)
    plt.ylabel("概率", fontsize=12)
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    prob = topic_distribution[keyword_dict[word], :]
    plt.bar(topic, prob)
    plt.title(word + "的主题分布", fontsize=14)
    plt.show()


if __name__ == "__main__":
    read_path = "./result/key_phrase_salience_rank.xlsx"
    topic_keyword = get_topic_and_keyword(read_path)
    topics, keywords, keyword_topic = keyword_statistic(topic_keyword)

    _, _, topic_keyword = topic_statistic(topic_keyword)
    topic_topk_keyword = get_topic_topk_words(topic_keyword)
    show_topic_topk_words(topic_topk_keyword)

    topic_distribution, keyword_dict = get_topic_distribution(keyword_topic, topics)
    root_path = read_path[:-5]
    save_pickle(topic_distribution, root_path + "_topic_distribution.pickle")
    save_pickle(keyword_dict, root_path + "_vocab_index.pickle")
    # print(topic_distribution[:5])

    # draw_topic_distribution("专家", topic_distribution, keyword_dict, topics)

    # 展示top6 keyword分布图
    keywords_top6 = Counter(keywords).most_common(6)
    print("Top 6 words: ", keywords_top6)

    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.95, hspace=0.35, wspace=0.3)

    topic = list(topics.keys())
    index = 1
    for keyword, _ in keywords_top6:
        plt.subplot(2, 3, index)
        # 设置坐标标签标注和字体大小
        plt.xlabel("主题", fontsize=12)
        plt.ylabel("概率", fontsize=12)
        # 设置坐标刻度字体大小
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        # 添加网格显示
        plt.grid(linestyle="-", alpha=0.5, linewidth=1.5)
        prob = topic_distribution[keyword_dict[keyword], :]
        plt.bar(topic, prob)
        plt.title(keyword + "的主题分布", fontsize=14)
        index += 1
    plt.savefig(root_path + "_topic_distribution.png")
    plt.show()



