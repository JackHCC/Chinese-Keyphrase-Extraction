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
from ranks import text_rank, SG_rank, position_rank, expand_rank, tr, tpr, single_tpr, salience_rank, embed_rank, \
    SIF_rank

from utils import write_to_excel, get_runtime

algorithms = {"text_rank": 0, "SG_rank": 1, "position_rank": 2, "expand_rank": 3, "tr": 4, "tpr": 5, "single_tpr": 6,
              "salience_rank": 7, "embed_rank": 8, "SIF_rank": 9}
get_algorithms = {v: k for k, v in algorithms.items()}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="salience_rank", type=str,
                        help="Choose an algorithm in text_rank, SG_rank, position_rank, expand_rank, tr, tpr, single_tpr, salience_rank, embed_rank or SIF_rank")
    parser.add_argument("--data", default="./data/data.xlsx", type=str, help="Train dataset path")
    parser.add_argument("--topic_num", default=8, type=int, help="Topic numbers")
    parser.add_argument("--top_k", default=15, type=int, help="Keyphrase Extraction Number")
    parser.add_argument("--alpha", default=0.3, type=float,
                        help="A hyperparameter for salience_rank algorithm, between 0 and 1")
    parser.add_argument("--lambda_", default=0.85, type=float,
                        help="A hyperparameter for PageRank, between 0 and 1")
    parser.add_argument("--window_size", default=6, type=int,
                        help="Co-occurrence window size of co-occurrence matrix")
    parser.add_argument("--max_d", default=0.75, type=float,
                        help="Maximum distance of flat clusters from the hierarchical clustering")
    args = parser.parse_args()
    return args


def algorithm_switch(arg, topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, alpha, lambda_,
                     window_size, max_d):
    if arg == 0:
        return text_rank(txt, lambda_)
    elif arg == 1:
        return SG_rank()
    elif arg == 2:
        return position_rank(txt, window_size, lambda_)
    elif arg == 3:
        return expand_rank()
    elif arg == 4:
        return tr(txt, max_d, lambda_)
    elif arg == 5:
        return tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, lambda_)
    elif arg == 6:
        return single_tpr(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, lambda_)
    elif arg == 7:
        return salience_rank(topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, txt, article_id, alpha,
                             lambda_)
    elif arg == 8:
        return embed_rank(txt)
    elif arg == 9:
        return SIF_rank()


@get_runtime
def main(arg, data_path, topic_num, top_k, alpha, lambda_, window_size, max_d):
    data, tf_feature_names, topic_x_word_matrix, docx_x_topic_matrix = get_matrix(topic_num, data_path)
    cut_data = data.content_cut
    all_phrases = []
    for article_id, text in enumerate(cut_data):
        text = text.strip()
        text = text.split(' ')
        phrases = algorithm_switch(arg, topic_x_word_matrix, docx_x_topic_matrix, tf_feature_names, text, article_id,
                                   alpha, lambda_, window_size, max_d)
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
    main(algorithms[args.alg], args.data, args.topic_num, args.top_k, args.alpha, args.lambda_, args.window_size,
         args.max_d)
