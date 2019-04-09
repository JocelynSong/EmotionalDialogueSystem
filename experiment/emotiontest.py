import sys
import os

import numpy as np

import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

__author__ = "Song"


emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}


def compute_emotion_scores(filename):
    scores = np.zeros([6, 2], dtype=float)
    count_scores = np.zeros([6, 6], dtype=float)
    print(scores)

    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    for i in range(200):
        post_ind = i * 15
        start_ind = post_ind + 7
        end_ind = post_ind + 13
        for j in range(start_ind, end_ind):
            line = lines[j]
            lemmas = line.strip().split()
            content_score = int(lemmas[0].strip())
            emotion_score = int(lemmas[1].strip())
            index = j - start_ind
            scores[index][0] = scores[index][0] + content_score
            scores[index][1] = scores[index][1] + emotion_score

            if content_score == 2 and emotion_score == 1:
                count_scores[index][0] = count_scores[index][0] + 1
            elif content_score == 1 and emotion_score == 1:
                count_scores[index][1] = count_scores[index][1] + 1
            elif content_score == 0 and emotion_score == 1:
                count_scores[index][2] = count_scores[index][2] + 1
            elif content_score == 2 and emotion_score == 0:
                count_scores[index][3] = count_scores[index][3] + 1
            elif content_score == 1 and emotion_score == 0:
                count_scores[index][4] = count_scores[index][4] + 1
            elif content_score == 0 and emotion_score == 0:
                count_scores[index][5] = count_scores[index][5] + 1

    for i in range(6):
        scores[i][0] = float(scores[i][0]) / float(200)
        scores[i][1] = float(scores[i][1]) / float(200)
        print("emotion=%s, content score = %f, emotion score = %f\n" % (id2emotion[i], scores[i][0], scores[i][1]))

        print("Ratio of %s:" % id2emotion[i])
        print("2-1: %f\n 1-1: %f\n 0-1: %f\n 2-0: %f\n 1-0: %f\n 0-0: %f\n" % (
               count_scores[i][0], count_scores[i][1], count_scores[i][2], count_scores[i][3], count_scores[i][4],
               count_scores[i][5]))

    overall_score = np.average(scores, axis=0)
    print("overall score: content=%f, emotion=%f\n" % (overall_score[0], overall_score[1]))


def compute_content_emotion_class(filename):
    scores = np.zeros([6], dtype=float)

    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    for i in range(200):
        post_ind = i * 15
        start_ind = post_ind + 7
        end_ind = post_ind + 12
        for j in range(start_ind, end_ind):
            line = lines[j]
            lemmas = line.strip().split()
            content_score = int(lemmas[0].strip())
            emotion_score = int(lemmas[1].strip())
            if content_score == 2 and emotion_score == 1:
                scores[0] = scores[0] + 1
            elif content_score == 1 and emotion_score == 1:
                scores[1] = scores[1] + 1
            elif content_score == 0 and emotion_score == 1:
                scores[2] = scores[2] + 1
            elif content_score == 2 and emotion_score == 0:
                scores[3] = scores[3] + 1
            elif content_score == 1 and emotion_score == 0:
                scores[4] = scores[4] + 1
            elif content_score == 0 and emotion_score == 0:
                scores[5] = scores[5] + 1

    total_score = scores[0] + scores[1] + scores[2] + scores[3] + scores[4] + scores[5]

    print("Results of this model:\n2-1: %f\n 1-1: %f\n 0-1: %f\n 2-0: %f\n 1-0: %f\n 0-0: %f\n" % (scores[0], scores[1],
                                                                                                   scores[2], scores[3],
                                                                                                   scores[4],
                                                                                                   scores[5]))

    for i in range(6):
        scores[i] = scores[i] / total_score

    print("Results of this model:\n2-1: %f\n 1-1: %f\n 0-1: %f\n 2-0: %f\n 1-0: %f\n 0-0: %f\n" % (scores[0], scores[1],
                                                                                                   scores[2], scores[3],
                                                                                                   scores[4], scores[5]))


def compute_generation_probability(true_label_file, predict_label_file, write_file):
    f_true = open(true_label_file, "r", encoding="utf-8")
    f_predict = open(predict_label_file, "r", encoding="utf-8")
    true_lines = f_true.readlines()
    pred_lines = f_predict.readlines()
    f_true.close()
    f_predict.close()

    total_length = len(true_lines)
    pred_lines = pred_lines[: total_length]

    matrix = np.zeros([6, 6], dtype=float)
    for t_line, p_line in zip(true_lines, pred_lines):
        t_label = emotion_dict[t_line.strip()]
        p_label = emotion_dict[p_line.strip()]
        matrix[t_label][p_label] += 1.0

    label_sum = np.sum(matrix, axis=1, keepdims=True)
    matrix = matrix / label_sum

    f = open(write_file, "w", encoding="utf-8")
    for t in matrix:
        for data in t:
            f.write(str(data))
            f.write(" ")
        f.write("\n")
    f.close()


def read_probability(file_name):
    f = open(file_name, "r", encoding="utf-8")
    matrix = []
    for line in f.readlines():
        data = []
        lemmas = line.strip().split()
        for lemma in lemmas:
            data.append(float(lemma))
        matrix.append(data)
    return matrix


def visualization(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    # fmt = '.2f' if normalize else 'd'
    fmt = ".4f"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted Emotion Category', fontsize=20)
    plt.xlabel('Desired Emotion Category', fontsize=20)


if __name__ == "__main__":
    """
    data_dir = "E:\\NLP\\code\\dialogue\\experiment\\data\\emotionTestData"
    data_file = os.path.join(data_dir, "model2.emotion.test.txt")

    compute_content_emotion_class(data_file)

    """
    data_dir = "E:\\NLP\\code\\dialogue\\experiment\\data\\visualization"
    write_file = os.path.join(data_dir, "ecm.label.txt")

    matrix = np.array(read_probability(write_file))
    np.set_printoptions(precision=4)
    class_names = ["anger", "disgust", "contentment", "joy", "sadness", "neutral"]

    plt.figure()
    visualization(matrix, classes=class_names, title='')

    plt.show()

