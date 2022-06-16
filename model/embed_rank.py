#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Chinese-Keyphrase-Extraction 
@File    :embed_rank.py
@Author  :JackHCC
@Date    :2022/6/16 21:11 
@Desc    :EmbedRank Model

'''
import re
import os
import jieba
import jieba.posseg as jp
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


NUMBERS_PATTERN = re.compile(r'[0-9]+.?')

INVALID_PATTERN = re.compile(r'^\W+$')

POS_DICT = os.path.join("./config", 'POS_dict.txt')
JIEBA_USER_DICT = os.path.join("./config", 'jieba_user_dict.txt')

# 加载词性筛选配置
FLAG_LIST = []
with open(POS_DICT, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        FLAG_LIST.append(line)

jieba.load_userdict(JIEBA_USER_DICT)
jieba.initialize()


class Doc2VecEmbedRank(object):
    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path)

    def extract_keyword(self, document, _lambda=0.5):
        phrase_ids, phrases, similarities = self._mmr(document, _lambda)
        if len(phrases) == 0 or len(phrase_ids) == 0 or len(similarities) == 0:
            return []

        outputs = set()
        for idx in phrase_ids:
            outputs.add((phrases[idx], similarities[idx][0]))
        outputs = list(outputs)
        return outputs

    def _mmr(self, document, _lambda=0.5):
        tokens = self._tokenize(document)
        if len(tokens) == 0:
            return [], [], []
        document_embedding = self.model.infer_vector(tokens)
        phrases, phrase_embeddings = self._create_phrases_with_embeddings(document)
        if len(phrases) == 0:
            return [], [], []
        phrase_embeddings = np.array(phrase_embeddings)  # shape (num_phrases, embedding_size)
        N = len(phrases)
        # similarity between each phrase and document
        phrase_document_similarities = cosine_similarity(phrase_embeddings, document_embedding.reshape(1, -1))
        # similarity between phrases
        phrase_phrase_similarities = cosine_similarity(phrase_embeddings)

        # MMR
        # 1st iteration
        unselected = list(range(len(phrases)))
        select_idx = np.argmax(phrase_document_similarities)  # most similiar phrase of document

        selected = [select_idx]
        unselected.remove(select_idx)

        # other iterations
        for _ in range(N - 1):
            mmr_distance_to_doc = phrase_document_similarities[unselected, :]
            mmr_distance_between_phrases = np.max(phrase_phrase_similarities[unselected][:, selected], axis=1)

            mmr = _lambda * mmr_distance_to_doc - (1 - _lambda) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected, phrases, phrase_document_similarities

    def _create_phrases_with_embeddings(self, document):
        phrases = []
        embeddings = []
        for w, pos in jp.lcut(document):
            if any(p in pos for p in FLAG_LIST):
                phrases.append(w)
                vector = self.model.infer_vector([w])
                embeddings.append(vector)
        return phrases, embeddings

    def _tokenize(self, document):
        tokens = []
        for w in jieba.cut(document):
            w = w.strip()
            if not w:
                continue
            if INVALID_PATTERN.match(w):
                continue
            tokens.append(w)
        return tokens
