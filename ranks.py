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
from collections import Counter, defaultdict
import math
import copy
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

from model.embed_rank import Doc2VecEmbedRank

from utils import set_graph_edges


def text_rank(text, lambda_):
    """

    @param text: list，分词后的文章输入
    @param lambda_: float，PageRank参数，0-1之间
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    # score nodes using default pagerank algorithm
    ranks = networkx.pagerank(graph, lambda_)

    sorted_phrases = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases


def SG_rank():
    raise NotImplemented


def position_rank(text, window_size, lambda_):
    """

    @param text: list，分词后的文章输入
    @param window_size: int，共现窗口大小
    @param lambda_: float，PageRank参数，0-1之间
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    def weight_total(matrix, idx, s_vec):
        """Sum weights of adjacent nodes.

        Choose 'j'th nodes which is adjacent to 'i'th node.
        Sum weight in 'j'th column, then devide wij(weight of index i,j).
        This calculation is applied to all adjacent node, and finally return sum of them.

        """
        return sum([(wij / matrix.sum(axis=0)[j]) * s_vec[j] for j, wij in enumerate(matrix[idx]) if not wij == 0])

    unique_word_list = set([word for word in words])
    n = len(unique_word_list)

    adjancency_matrix = np.zeros((n, n))
    word2idx = {w: i for i, w in enumerate(unique_word_list)}
    p_vec = np.zeros(n)
    # store co-occurence words
    co_occ_dict = {w: [] for w in unique_word_list}

    # 1. initialize  probability vector
    for i, w in enumerate(words):
        # add position score
        p_vec[word2idx[w]] += float(1 / (i + 1))
        for window_idx in range(1, math.ceil(window_size / 2) + 1):
            if i - window_idx >= 0:
                co_list = co_occ_dict[w]
                co_list.append(words[i - window_idx])
                co_occ_dict[w] = co_list

            if i + window_idx < len(words):
                co_list = co_occ_dict[w]
                co_list.append(words[i + window_idx])
                co_occ_dict[w] = co_list

    # 2. create adjancency matrix from co-occurence word
    for w, co_list in co_occ_dict.items():
        cnt = Counter(co_list)
        for co_word, freq in cnt.most_common():
            adjancency_matrix[word2idx[w]][word2idx[co_word]] = freq

    adjancency_matrix = adjancency_matrix / adjancency_matrix.sum(axis=0)
    p_vec = p_vec / p_vec.sum()
    # principal eigenvector s
    s_vec = np.ones(n) / n

    # threshold
    lambda_val = 1.0
    loop = 0
    # compute final principal eigenvector
    while lambda_val > 0.001:
        next_s_vec = copy.deepcopy(s_vec)
        for i, (p, s) in enumerate(zip(p_vec, s_vec)):
            next_s = (1 - lambda_) * p + lambda_ * (weight_total(adjancency_matrix, i, s_vec))
            next_s_vec[i] = next_s
        lambda_val = np.linalg.norm(next_s_vec - s_vec)
        s_vec = next_s_vec
        loop += 1
        if loop > 100:
            break

    # score original words and phrases
    ranks = {word: s_vec[word2idx[word]] for word in unique_word_list}
    sorted_phrases = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    return sorted_phrases


def expand_rank():
    raise NotImplemented


def tr(text, max_d, lambda_):
    words = text

    def calc_distance(topic_a, topic_b, position_map):
        """
        Calculate distance between 2 topics
        :param topic_a: list if phrases in a topic A
        :param topic_b: list if phrases in a topic B
        :return: int
        """
        result = 0
        for phrase_a in topic_a:
            for phrase_b in topic_b:
                if phrase_a != phrase_b:
                    phrase_a_positions = position_map[phrase_a]
                    phrase_b_positions = position_map[phrase_b]
                    for a, b in product(phrase_a_positions, phrase_b_positions):
                        result += 1 / abs(a - b)
        return result

    position_map = defaultdict(list)

    # get position info
    for idx, word in enumerate(words):
        position_map[word].append(idx)

    # use term freq to convert phrases to vectors for clustering
    count = CountVectorizer()
    bag = count.fit_transform(list(position_map.keys()))

    # apply HAC
    Z = linkage(bag.toarray(), 'average')

    # identify clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    cluster_data = defaultdict(list)
    for n, cluster in enumerate(clusters):
        cluster_data[cluster].append(' '.join(sorted([str(i) for i in count.inverse_transform(bag.toarray()[n])[0]])))
    topic_clusters = [frozenset(i) for i in cluster_data.values()]

    topic_graph = networkx.Graph()
    topic_graph.add_weighted_edges_from(
        [(v, u, calc_distance(v, u, position_map)) for v in topic_clusters for u in topic_clusters if u != v])
    ranks = networkx.pagerank(topic_graph, lambda_, weight='weight')

    # sort topic by rank
    topics = sorted([(b, list(a)) for a, b in ranks.items()], reverse=True)
    sorted_phrases = [(topic_list[0], score) for score, topic_list in topics]
    return sorted_phrases


def tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id, lambda_):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @param lambda_: float，PageRank参数，0-1之间
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
        ranks = networkx.pagerank(graph, lambda_, personalization)
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


def single_tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id, lambda_):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @param lambda_: float，PageRank参数，0-1之间
    @return: 该文档的关键词汇得分排序列表
    """
    words = text

    # set the graph edges
    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    pt_new_dim = docx_x_topic_matrix[article_id, :] / sum(
        docx_x_topic_matrix[article_id, :])  # topic distribution for one doc
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

    ranks = networkx.pagerank(graph, lambda_, personalization)

    sorted_phrases = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases


def salience_rank(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id, alpha, lambda_):
    """

    @param topic_x_word_matrix: matrix，主题词汇分布矩阵（归一化后的）
    @param docx_x_topic_matrix: matrix，文档主题分布矩阵（未归一化的）
    @param tf_feature_names: list，词汇字典
    @param text: list，分词后的文章输入
    @param article_id: int，文章遍历过程的编号
    @param alpha: float，salience_rank算法的参数，用于控制语料库特异性和话题特异性之间的权衡，取值位于0到1之间，越趋近于1，话题特异性越明显，越趋近于0，语料库特异性越明显
    @param lambda_: float，PageRank参数，0-1之间
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
    ranks = networkx.pagerank(graph, lambda_, personalization)

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


def embed_rank(text):
    """

    @param text: list，分词后的文章输入
    @return: 该文档的关键词汇得分排序列表
    """
    words = text
    model_path = './model/embed_rank/embed_rank_doc2vec.bin'
    model = Doc2VecEmbedRank(model_path)

    sorted_ranks = sorted(model.extract_keyword(words), key=lambda x: x[1], reverse=True)
    return sorted_ranks


def SIF_rank(text, plus=True, lambda_=0.8, elmo_layers_weight=[0.0, 1.0, 0.0]):
    import thulac
    from model.SIF_rank.Embedding import WordEmbeddings, SentEmbeddings
    from model.SIF_rank.sif_rank import SIFRank, SIFRank_plus

    words = text.replace("\n", "").replace("\t", "")

    # download from https://github.com/HIT-SCIR/ELMoForManyLangs
    model_file = r'./model/SIF_rank/zhs.model/'

    elmo = WordEmbeddings(model_file)
    sif = SentEmbeddings(elmo, weightfile_pretrain='./model/SIF_rank/dict.txt',
                         weightfile_finetune='./model/SIF_rank/dict.txt', lamda=lambda_)

    # download from http://thulac.thunlp.org/
    zh_model = thulac.thulac(model_path=r'./model/SIF_rank/thulac.models/', user_dict=r'./config/jieba_user_dict.txt')

    if plus:
        sorted_ranks = SIFRank_plus(words, sif, zh_model, elmo_layers_weight=elmo_layers_weight)

    else:
        sorted_ranks = SIFRank(words, sif, zh_model, elmo_layers_weight=elmo_layers_weight)

    return sorted_ranks
