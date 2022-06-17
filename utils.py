#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :NLP 
@File    :utils.py
@Author  :JackHCC
@Date    :2022/6/12 15:48 
@Desc    :

'''
from itertools import combinations as combinations
from queue import Queue
import time
import pickle as pkl

WINDOW_SIZE = 2


def get_first_window(split_text):
    return split_text[:WINDOW_SIZE]


# tokens is a list of words
def set_graph_edge(graph, tokens, word_a, word_b):
    if word_a in tokens and word_b in tokens:
        edge = (word_a, word_b)
        if graph.has_node(word_a) and graph.has_node(word_b) and not graph.has_edge(*edge):
            graph.add_edge(*edge)


def process_first_window(graph, tokens, split_text):
    first_window = get_first_window(split_text)
    for word_a, word_b in combinations(first_window, 2):
        set_graph_edge(graph, tokens, word_a, word_b)


def init_queue(split_text):
    queue = Queue()
    first_window = get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def queue_iterator(queue):
    iterations = queue.qsize()
    for i in range(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def process_word(graph, tokens, queue, word):
    for word_to_compare in queue_iterator(queue):
        set_graph_edge(graph, tokens, word, word_to_compare)


def update_queue(queue, word):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)


def process_text(graph, tokens, split_text):
    queue = init_queue(split_text)
    for i in range(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        process_word(graph, tokens, queue, word)
        update_queue(queue, word)


def set_graph_edges(graph, tokens, split_text):
    process_first_window(graph, tokens, split_text)
    process_text(graph, tokens, split_text)


def write_to_excel(obj, save_path):
    obj.to_excel(save_path)


def get_runtime(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = fn(*args, **kwargs)
        end_time = time.time()
        print("程序运行时间: {} s".format(end_time - start_time))
        return results
    return wrapper


def save_pickle(obj, path):
    file = open(path, 'wb')
    pkl.dump(obj, file)


def read_pickle(path):
    with open(path, 'rb') as file:
        obj = pkl.load(file)
    return obj

