import numpy as np
import os
import sys

from data_utils.prepare_dialogue_data import read_total_embeddings


def read_response_file(file_name, word2id):
    word_unk = "</s>"
    word_unk_id = word2id[word_unk]

    f = open(file_name, "r", encoding="utf-8")
    response_data = []
    for line in f.readlines():
        words = line.strip().split()
        words_indexes = [word2id[word] if word in word2id else word_unk_id for word in words]
        response_data.append(words_indexes)
    f.close()
    return response_data


def get_norm_embedding(embedding):
    norm = np.sqrt(np.sum(np.square(embedding)))
    new_embedding = embedding / norm
    return new_embedding


def get_norm_embeddings(embeddings):
    new_embeddings = []
    for embedding in embeddings:
        new_embedding = get_norm_embedding(embedding)
        new_embeddings.append(new_embedding)
    return new_embeddings


def compute_greedy_metric(generate_response_data, true_response_data, embeddings):
    def one_data_similarity(one_line_generate, one_line_true):
        scores = 0.0
        for index in one_line_generate:
            cos_scores = []
            for new_index in one_line_true:
                this_score = np.dot(embeddings[index], embeddings[new_index])
                cos_scores.append(this_score)
            scores += np.max(cos_scores)
        score = scores / len(one_line_generate)
        return score

    results = []
    for data_generation, data_true in zip(generate_response_data, true_response_data):
        if len(data_generation) == 0 or len(data_true) == 0:
            results.append(0.0)
            continue

        score1 = one_data_similarity(data_generation, data_true)
        score2 = one_data_similarity(data_true, data_generation)
        results.append((score1 + score2) / 2.0)
    final_score = np.sum(np.array(results)) / len(results)
    print("greedy score=%f\n" % final_score)


def compute_average_metric(generate_response_data, true_response_data, embeddings, dimension):
    def compute_sentence_embedding(one_line_generate):
        total_embedding = np.zeros([dimension])
        for index in one_line_generate:
            total_embedding = total_embedding + embeddings[index]
        total_embedding = get_norm_embedding(total_embedding)
        return total_embedding

    results = []
    for data_generation, data_true in zip(generate_response_data, true_response_data):
        if len(data_generation) == 0 or len(data_true) == 0:
            results.append(0.0)
            continue

        first_new_emb = compute_sentence_embedding(data_generation)
        second_new_emb = compute_sentence_embedding(data_true)
        results.append(np.dot(first_new_emb, second_new_emb))
    final_score = np.sum(np.array(results)) / len(results)
    print("average score=%f\n" % final_score)


def compute_extreme_metric(generate_response_data, true_response_data, embeddings, dimension):
    def compute_sentence_embedding(one_line_generate):
        total_embeddings = []
        for index in one_line_generate:
            total_embeddings.append(embeddings[index])
        total_embeddings = np.array(total_embeddings)
        final_embedding_max = np.max(total_embeddings, axis=0)
        final_embedding_min = np.min(total_embeddings, axis=0)
        final_embedding = final_embedding_max
        for i in range(dimension):
            if np.abs(final_embedding_max[i]) < np.abs(final_embedding_min[i]):
                final_embedding[i] = final_embedding_min[i]
        final_embedding = get_norm_embedding(final_embedding)
        return final_embedding

    results = []
    for data_generation, data_true in zip(generate_response_data, true_response_data):
        if len(data_generation) == 0 or len(data_true) == 0:
            results.append(0.0)
            continue

        first_new_emb = compute_sentence_embedding(data_generation)
        second_new_emb = compute_sentence_embedding(data_true)
        results.append(np.dot(first_new_emb, second_new_emb))
    final_score = np.sum(np.array(results)) / len(results)
    print("extreme score=%f\n" % final_score)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("embedding_metrics.py")))
    data_path = os.path.join(model_path, "data")

    embedding_file = os.path.join(data_path, "embedding\\emotion.embeddings.txt")

    true_response_file = os.path.join(data_path, "stc_data\\train_test\\test.response.filter.txt")

    new_data_dir = "E:\\NLP\\code\\dialogue\\experiment\\data\\bleu&distinct"
    test_response_file = os.path.join(new_data_dir, "EmotionDS\\generation.data.txt")

    max_vocab_size = 100000
    dimension = 100

    print("read embedding")
    embeddings, word2id, word_list = read_total_embeddings(embedding_file, max_vocab_size)
    embeddings = get_norm_embeddings(embeddings)

    print("read response data")
    generate_response = read_response_file(test_response_file, word2id)
    true_response = read_response_file(true_response_file, word2id)

    print("compute metrics")
    compute_average_metric(generate_response, true_response, embeddings, dimension)
    compute_greedy_metric(generate_response, true_response, embeddings)
    compute_extreme_metric(generate_response, true_response, embeddings, dimension)

