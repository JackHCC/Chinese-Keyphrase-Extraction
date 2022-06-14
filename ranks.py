#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :NLP 
@File    :ranks.py
@Author  :JackHCC
@Date    :2022/6/12 15:01 
@Desc    :Rank algorithms

'''
import numpy as np
from numpy import linalg
import networkx

from utils import set_graph_edges


def text_rank(text):
    """

    @param text: list，分词后的文章输入
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    # score nodes using default pagerank algorithm
    ranks = networkx.pagerank(graph)

    sorted_phrases = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases


def tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    col_sums = topic_x_word_matrix.sum(axis=0)
    p_tw = topic_x_word_matrix / col_sums[np.newaxis, :]  # normalize column-wise: each word (col) is a distribution

    # run page rank for each topic
    phrases_scores = {}  # keyphrse: list of ranks for each topic
    for t in range(len(topic_x_word_matrix)):
        # construct personalization vector and run PR
        personalization = {}
        idx = 0
        for n, _ in list(graph.nodes(data=True)):
            if n in sorted(tf_feature_names):
                if n in words:
                    personalization[n] = p_tw[t, idx]
                else:
                    personalization[n] = 0
            else:
                personalization[n] = 0
            idx = idx + 1
        ranks = networkx.pagerank(graph, 0.85, personalization)
        for word, score in ranks.items():
            if word in phrases_scores:
                phrases_scores[word].append(score)
            else:
                phrases_scores[word] = [score]

    for word, scores in phrases_scores.items():
        phrases_scores[word] = np.dot(np.array(scores),
                                      docx_x_topic_matrix[article_id, :] / sum(docx_x_topic_matrix[article_id, :]))

    sorted_phrases = sorted(phrases_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases


def single_tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    # set the graph edges
    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    pt_new_dim = docx_x_topic_matrix[article_id, :] / sum(docx_x_topic_matrix[article_id, :])  # topic distribution for one doc
    pt_new_dim = pt_new_dim[None, :]
    weights = np.dot(topic_x_word_matrix.T, pt_new_dim.T)
    weights = weights / linalg.norm(pt_new_dim, 'fro')  # cos similarity normalization

    personalization = {}
    count = 0
    for n, _ in list(graph.nodes(data=True)):
        if n in sorted(tf_feature_names):
            if n in words:
                personalization[n] = weights[count] / (linalg.norm(pt_new_dim, 'fro') * linalg.norm(
                    topic_x_word_matrix[:, count]))  # cos similarity normalization
            else:
                personalization[n] = 0
        else:
            personalization[n] = 0
        count = count + 1

    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    factor = 1.0 / sum(personalization.values())  # normalize the personalization vec

    for k in personalization:
        personalization[k] = personalization[k] * factor

    ranks = networkx.pagerank(graph, 0.85, personalization)

    sorted_phrases = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases


def salience_rank(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id, alpha=0.3):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @param alpha: float，salience_rank算法的参数，用于控制语料库特异性和话题特异性之间的权衡，取值位于0到1之间，越趋近于1，话题特异性越明显，越趋近于0，语料库特异性越明显
    @return: 该文档的关键词汇得分排序列表
    """
    # word输入的该id文档中获取的有效词语
    words = text
    # set the graph edges
    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    col_sums = topic_x_word_matrix.sum(axis=0)
    # pw = col_sums / np.sum(col_sums)

    p_tw = topic_x_word_matrix / col_sums[np.newaxis, :]  # normalize column-wise: each word (col) is a distribution
    pt_new_dim = docx_x_topic_matrix[article_id, :] / sum(docx_x_topic_matrix[article_id, :])
    pt_new_dim = pt_new_dim[None, :]
    p_tw_by_pt = np.divide(p_tw, pt_new_dim.T)  # divide each column by the vector pt elementwise
    kernel = np.multiply(p_tw, np.log(p_tw_by_pt))
    distinct = kernel.sum(axis=0)
    distinct = (distinct - np.min(distinct)) / (np.max(distinct) - np.min(distinct))  # normalize

    personalization = {}
    count = 0
    for n, _ in list(graph.nodes(data=True)):
        if n in sorted(tf_feature_names):
            if n in words:
                personalization[n] = (1.0 - alpha) * sum(topic_x_word_matrix[:, count]) + alpha * distinct[count]
            else:
                personalization[n] = 0
        else:
            personalization[n] = 0
        count = count + 1

    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph, 0.85, personalization)

    # Paper: https://aclanthology.org/P17-2084/
    # ranks = networkx.pagerank(graph, 0.95, None, 1, 5.0e-1)
    # scores = {}
    # lamb = 0.7
    # assert len(ranks) == len(personalization)
    # for key, value in ranks.items():
    #     scores[key] = lamb * ranks[key] + (1 - lamb) * personalization[key]
    # ranks = scores

    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_ranks
