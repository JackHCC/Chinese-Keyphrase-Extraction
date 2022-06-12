#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :NLP 
@File    :lda.py
@Author  :JackHCC
@Date    :2022/6/12 14:32 
@Desc    :Latent Dirichlet Allocation

'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt

from process import data_word_cut


def print_top_words(topic_x_word_matrix, feature_names, n_top_words=25):
    """
    打印每个主题下的关键的词语
    @param topic_x_word_matrix:
    @param feature_names:
    @param n_top_words:
    @return:
    """
    top_words = []
    for topic_idx, topic in enumerate(topic_x_word_matrix):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        top_words.append(topic_w)
        print(topic_w)
    return top_words


def pred_docx_topic(data, docx_x_topic_matrix):
    """
    根据每个文档预测其主题并写入xlsx表格中
    @param data:
    @param docx_x_topic_matrix:
    """
    topic = []
    for t in docx_x_topic_matrix:
        topic.append(list(t).index(np.max(t)))
    data['topic'] = topic
    data.to_excel("./result/data_lda_topic.xlsx", index=False)


def get_best_topic_num(tf, n_max_topics=16):
    """
    根据困惑度获取最优的主题数目
    @param tf:
    @param n_max_topics:
    """
    plexs = []
    scores = []
    for i in range(1, n_max_topics):
        print(i)
        lda_model = LatentDirichletAllocation(n_components=i, max_iter=100,
                                              learning_method='batch',
                                              learning_offset=50, random_state=0)
        lda_model.fit(tf)
        plexs.append(lda_model.perplexity(tf))
        scores.append(lda_model.score(tf))

    n_t = n_max_topics - 1
    x = list(range(1, n_t))
    plt.plot(x, plexs[1:n_t])
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.show()


def get_vector(data, stop_words='english', max_df=0.5, min_df=10):
    tf_vector = CountVectorizer(strip_accents='unicode',
                                stop_words=stop_words,
                                max_df=max_df,
                                min_df=min_df)
    tf = tf_vector.fit_transform(data.content_cut)
    tf_feature_names = tf_vector.get_feature_names()
    return tf, tf_feature_names


def lda(tf, topic_num=8, max_iter=100, doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0):
    lda_model = LatentDirichletAllocation(n_components=topic_num, max_iter=max_iter,
                                          learning_method='batch',
                                          learning_offset=50,
                                          doc_topic_prior=doc_topic_prior,
                                          topic_word_prior=topic_word_prior,
                                          random_state=random_state)
    lda_model.fit(tf)
    # topic_x_word_matrix = lda_model.components_
    # normal lda_model.components_
    topic_x_word_matrix = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    docx_x_topic_matrix = lda_model.transform(tf)
    return topic_x_word_matrix, docx_x_topic_matrix


def get_matrix(topic_num, data_path="./data/data.xlsx"):
    """
    获取文档主题分布矩阵和主题词语分布矩阵
    @param topic_num:
    @param data_path:
    @return:
    """
    data = data_word_cut(data_path)
    tf, tf_feature_names = get_vector(data)
    topic_x_word_matrix, docx_x_topic_matrix = lda(tf, topic_num)
    print("word dict size: ", len(tf_feature_names))
    return data, tf_feature_names, topic_x_word_matrix, docx_x_topic_matrix


if __name__ == "__main__":
    cut_data = data_word_cut()
    tf, feature_name = get_vector(cut_data)
    topic_x_word, docx_x_topic = lda(tf)
    print_top_words(topic_x_word, feature_name)
    pred_docx_topic(cut_data, docx_x_topic)

    get_best_topic_num(tf)
