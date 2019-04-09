import sys
import os

from data_utils.prepare_dialogue_data import read_emotion_label, get_word_count, read_emotion_words


emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}


def get_bigrams(words):
    length = len(words)
    this_bigrams = list()
    for i in range(length - 1):
        this_bigrams.append(words[i: i+2])
    return this_bigrams


def compute_distinct_metric(filename):
    total_unigram, total_bigram = 0, 0
    unigrams, bigrams = list(), list()

    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        total_unigram += len(words)
        words_diff = set(words)
        for word in words_diff:
            if word not in unigrams:
                unigrams.append(word)
        this_bigrams = get_bigrams(words)
        total_bigram += len(this_bigrams)
        for bigram in this_bigrams:
            if bigram not in bigrams:
                bigrams.append(bigram)
    num_unigram = len(unigrams)
    num_bigram = len(bigrams)
    distinct_1 = float(num_unigram) / total_unigram
    distinct_2 = float(num_bigram) / total_bigram
    print("distinct one=%f, distinct two=%f\n" % (distinct_1, distinct_2))


def change_file_format(file1, file2):
    start_symbol = "<ss>"
    end_symbol = "<es>"

    f1 = open(file1, "r", encoding="utf-8")
    f2 = open(file2, "w", encoding="utf-8")
    for line in f1.readlines():
        words = line.strip().split()
        words = words[1:]
        if start_symbol in words:
            start_index = words.index(start_symbol) + 1
        else:
            start_index = 0
        if end_symbol in words:
            end_index = words.index(end_symbol)
        else:
            end_index = len(words)

        selected_words = words[start_index: end_index]
        sentence = " ".join(selected_words)
        f2.write(sentence)
        f2.write("\n")
    f1.close()
    f2.close()


def construct_vocab(emotion_word_dict, emotion_word_num):
    total_emotion_word = list()
    for i in range(6):
        emotion_word_lists = emotion_word_dict[id2emotion[i]]
        sorted_emotion_words = sorted(emotion_word_lists.items(), key=lambda x: x[1], reverse=True)

        length = 0
        index = 0
        while length < emotion_word_num:
            if sorted_emotion_words[index][0] not in total_emotion_word:
                total_emotion_word.append(sorted_emotion_words[index][0])
                length += 1
            index += 1
    return total_emotion_word


def read_test_data(file_name):
    f = open(file_name, "r", encoding="utf-8")
    responses = []
    for line in f.readlines():
        words = line.strip().split()
        responses.append(words)
    return responses


def is_valid_emotion(response, emotion_words):
    for word in response:
        if word in emotion_words:
            return True
    return False


def compute_ratio(test_responses, total_emotion_words, emotion_labels, emotion_num):
    count = 0
    total_num = 0
    for response, label in zip(test_responses, emotion_labels):
        start_ind = label * emotion_num
        end_ind = (label + 1) * emotion_num
        emotion_words = total_emotion_words[start_ind: end_ind]
        if is_valid_emotion(response, emotion_words):
            count += 1
    ratio = float(count) / float(len(test_responses))
    print("success ratio=%f\n" % ratio)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("automatic_evaluation.py")))
    data_path = os.path.join(model_path, "data")

    data_dir = "E:\\NLP\\code\\dialogue\\emotional chat machine\\exp\\test"
    response_file = os.path.join(data_dir, "running\\test.response9.txt")

    new_data_dir = "E:\\NLP\\code\\dialogue\\experiment\\data\\bleu&distinct\\EmotionDS"
    new_response_file = os.path.join(new_data_dir, "generation.data.txt")
    
    change_file_format(response_file, new_response_file)
    compute_distinct_metric(new_response_file)

    """
    emotion_word_count_file = os.path.join(data_path, "emotion_words_human\\word.count.txt")
    emotion_words_dir = os.path.join(data_path, "emotion_words_human")
    emotion_label_file = os.path.join(data_path, "stc_data\\train_test\\test.label.lstm.filter.txt")

    word_count = 5
    emotion_vocab_size = 200

    pre_word_count = get_word_count(emotion_word_count_file, word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    emotion_vocab = construct_vocab(emotion_words_dict, emotion_vocab_size)

    test_responses = read_test_data(response_file)
    test_emotion_labels = read_emotion_label(emotion_label_file)

    compute_ratio(test_responses, emotion_vocab, test_emotion_labels, emotion_vocab_size)
    """



