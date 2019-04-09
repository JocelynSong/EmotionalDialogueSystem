import os
import random
import numpy as np


emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}


def read_label(filename):
    labels = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        labels.append(emotion_dict[line.strip()])
    f.close()
    return labels


def compute_statistics(filename):
    labels = read_label(filename)
    valid_len = int(len(labels) * 0.05)
    train_len = len(labels) - valid_len * 2
    random.shuffle(labels)

    train_labels = labels[: train_len]
    valid_labels = labels[train_len: (train_len + valid_len)]
    test_labels = labels[(train_len + valid_len):]

    print("train=%d, valid=%d, test=%d\n" % (len(train_labels), len(valid_labels), len(test_labels)))

    numbers = np.zeros([6], dtype=int)
    for label in labels:
        numbers[label] += 1

    for i in range(6):
        print("%s=%d\n" % (id2emotion[i], numbers[i]))


if __name__ == "__main__":
    data_dir = "E:\\NLP\\code\\dialogue\\emotional chat machine\\data\\stc_data"
    file_name = "stc.labels.txt"
    file_path = os.path.join(data_dir, file_name)

    compute_statistics(file_path)
