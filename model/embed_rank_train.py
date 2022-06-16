#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Chinese-Keyphrase-Extraction 
@File    :embed_rank_train.py
@Author  :JackHCC
@Date    :2022/6/16 23:53 
@Desc    :Train Doc2Vec to get Embedding Matrix

'''
import gensim
import pandas as pd
import jieba
import re
import os
import jieba.posseg as jp

INVALID_PATTERN = re.compile(r'^\W+$')

JIEBA_USER_DICT = os.path.join("../config", 'jieba_user_dict.txt')
jieba.load_userdict(JIEBA_USER_DICT)
jieba.initialize()


def tokenize(text):
    tokens = []
    for w, pos in jp.lcut(text.lower()):
        if not pos:
            continue
        if INVALID_PATTERN.match(w):
            continue
        if not w.strip():
            continue
        tokens.append(w.strip())
    return tokens


def read_corpus(fname):
    data = pd.read_excel(fname)
    for i, row in data.iterrows():
        text = row['content']
        tokens = tokenize(text.strip('\n').strip())
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def train(input_file, model_path, vocab_path, vector_size=100, min_count=5, works=8, epochs=10):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, max_vocab_size=100000, works=works)
    train_corpus = read_corpus(input_file)
    model.build_vocab(train_corpus)

    vocab_dir = '/'.join(str(vocab_path).split('/')[:-1])
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    with open(vocab_path, mode='wt', encoding='utf8') as f:
        for k, v in model.wv.vocab.items():
            f.write(k + '\t' + str(v.count) + '\n')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=epochs)
    model.save(model_path)


if __name__ == "__main__":
    # Train Doc2Vec to get Embedding Matrix
    data_path = "../data/data.xlsx"
    model_path = "./embed_rank/embed_rank_doc2vec.bin"
    vocab_path = "./embed_rank/embed_rank_doc2vec.txt"
    train(data_path, model_path, vocab_path, epochs=20)
