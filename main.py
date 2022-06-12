#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :NLP 
@File    :main.py
@Author  :JackHCC
@Date    :2022/6/12 17:34 
@Desc    :run main program

'''
import argparse

from lda import get_matrix
from ranks import text_rank, tpr, single_tpr, salience_rank

from utils import write_to_excel

algorithms = {"text_rank": 0, "tpr": 1, "salience_rank": 2, "single_tpr": 3}
get_algorithms = {0: "text_rank", 1: "tpr", 2: "salience_rank", 3: "single_tpr"}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="salience_rank", type=str,
                        help="Choose an algorithm in text_rank, tpr, single_tpr or salience_rank")
    parser.add_argument("--data", default="./data/data.xlsx", type=str, help="Train dataset path")
    parser.add_argument("--topic_num", default=8, type=int, help="Topic numbers")
    parser.add_argument("--top_k", default=15, type=int, help="Keyphrase Extraction Number")
    parser.add_argument("--alpha", default=0.3, type=float, help="A hyperparameter for salience_rank algorithm")
    args = parser.parse_args()
    return args


def algorithm_switch(arg, topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, alpha=0.3):
    if arg == 0:
        return text_rank(txt)
    elif arg == 1:
        return tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id)
    elif arg == 2:
        return salience_rank(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, alpha)
    elif arg == 3:
        return single_tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id)


def main(arg, data_path, topic_num=8, top_k=15, alpha=0.3):
    data, tf_feature_names, topic_x_word_matrix, docx_x_topic_matrix = get_matrix(topic_num, data_path)
    cut_data = data.content_cut
    all_phrases = []
    for article_id, text in enumerate(cut_data):
        text = text.strip()
        text = text.split(' ')
        # phrases = salience_rank(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id)
        phrases = algorithm_switch(arg, topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id,
                                   alpha)
        phrases_top_k = []
        tk = min(top_k, len(phrases))
        for i in range(tk):
            phrases_top_k.append(phrases[i][0])
        phrases_top_k = ';'.join(phrases_top_k)
        print(article_id, " : ", phrases_top_k)
        all_phrases.append(phrases_top_k)
    data['key_phrase'] = all_phrases
    # 写入excel
    write_to_excel(data, "./result/key_phrase_" + get_algorithms[arg] + ".xlsx")
    return all_phrases


if __name__ == "__main__":
    args = args()
    main(algorithms[args.alg], args.data, args.topic_num, args.top_k, args.alpha)
