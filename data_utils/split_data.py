import os
import sys
import random

import numpy as np


def split_train_test_data(post_data, response_data, label_origin, label_lstm):
    total_data = [[post, response, label1, label2] for post, response, label1, label2 in zip(post_data, response_data,
                                                                                             label_origin, label_lstm)]
    random.shuffle(total_data)

    total_len = len(post_data)
    train_len = int(total_len * 0.9)
    train_data = total_data[: train_len]
    test_data = total_data[train_len:]

    train_post_data = [data[0] for data in train_data]
    train_response_data = [data[1] for data in train_data]
    train_label_origin = [data[2] for data in train_data]
    train_label_lstm = [data[3] for data in train_data]
    test_post_data = [data[0] for data in test_data]
    test_response_data = [data[1] for data in test_data]
    test_label_origin = [data[2] for data in test_data]
    test_label_lstm = [data[3] for data in test_data]
    return (train_post_data, train_response_data, train_label_origin, train_label_lstm, test_post_data,
            test_response_data, test_label_origin, test_label_lstm)


def read_data(file_name):
    f = open(file_name, "r", encoding="utf-8")
    lines = f.readlines()
    data = []
    for line in lines:
        words = line.strip().split()
        data.append(words)
    return data


def read_label(file_name):
    f = open(file_name, "r", encoding="utf-8")
    labels = []
    for line in f.readlines():
        labels.append(line.strip())
    return labels


def filter_data(post_data, response_data, label_origin, label_lstm, max_len):
    new_post, new_response, new_label_origin, new_label_lstm = [], [], [], []
    for post, response, label1, label2 in zip(post_data, response_data, label_origin, label_lstm):
        if len(post) > max_len:
            continue
        new_post.append(post)
        new_response.append(response)
        new_label_origin.append(label1)
        new_label_lstm.append(label2)
    return new_post, new_response, new_label_origin, new_label_lstm


def write_data(filename, data):
    f = open(filename, "w", encoding="utf-8")
    for words in data:
        line = " ".join(words)
        f.write(line)
        f.write("\n")
    f.close()


def write_label(filename, labels):
    f = open(filename, "w", encoding="utf-8")
    for line in labels:
        f.write(line)
        f.write("\n")
    f.close()


def main(post_file, response_file, label_origin_file, label_lstm_file, data_dir):
    post_data = read_data(post_file)
    response_data = read_data(response_file)
    label_origin = read_label(label_origin_file)
    label_lstm = read_label(label_lstm_file)

    post_data, response_data, label_origin, label_lstm = filter_data(post_data, response_data, label_origin, label_lstm,
                                                                     10)
    post_file = os.path.join(data_dir, "test.post.filter.txt")
    response_file = os.path.join(data_dir, "test.response.filter.txt")
    label_origin_file = os.path.join(data_dir, "test.label.origin.filter.txt")
    label_lstm_file = os.path.join(data_dir, "test.label.lstm.filter.txt")
    write_data(post_file, post_data)
    write_data(response_file, response_data)
    write_label(label_origin_file, label_origin)
    write_label(label_lstm_file, label_lstm)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("split_data.py")))
    data_path = os.path.join(model_path, "data\\stc_data\\train_test")

    post_file = os.path.join(data_path, "test.post.txt")
    response_file = os.path.join(data_path, "test.response.txt")
    label_origin_file = os.path.join(data_path, "test.label.txt")
    label_lstm_file = os.path.join(data_path, "test.label.lstm")

    main(post_file, response_file, label_origin_file, label_origin_file, data_path)




