import os
import sys
import numpy as np

__author__ = "Jocelyn"

emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "none": 5}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}


def compute_content_scores(generate_responses, post_sen, stop_words):
    def compute_score_per_sen(res, post, stop):
        for word in res:
            if word in post and word not in stop:
                return 1.0
        return 0.0

    scores = list()
    for response in generate_responses:
        scores.append(compute_score_per_sen(response, post_sen, stop_words))
    return scores


def compute_emotion_scores(generate_responses, emotion_words):
    def compute_score_per_sen(res_sen, emotion_symbols):
        for word in res_sen:
            if word in emotion_symbols:
                return 2.0
        return 0.0

    scores = list()
    for response in generate_responses:
        score = compute_score_per_sen(response, emotion_words)
        scores.append(score)
    return scores


def compute_repetition_score(generate_responses, start_id, end_id):
    scores = list()
    for response in generate_responses:
        response = list(response)
        if start_id in response:
            start_index = response.index(start_id)
        else:
            start_index = 0
        if end_id in response:
            end_index = response.index(end_id)
        else:
            end_index = len(response)
        new_response = response[start_index + 1: end_index]
        score = len(new_response) - len(set(new_response))
        scores.append(score)
    return scores


def compute_emotion_candidates(generate_responses, emotion_words):
    def is_valid_emotion(res_sen, emotion_symbols):
        for word in res_sen:
            if word in emotion_symbols:
                return True
        return False

    candidates = list()
    for i in range(len(generate_responses)):
        if is_valid_emotion(generate_responses[i], emotion_words):
            candidates.append(i)
    if len(candidates) == 0:
        candidates = range(len(generate_responses))
    return candidates


def compute_content_candidates(generate_responses, candidates, post_sen, stop_words):
    def is_valid_content(res, post, stop):
        for word in res:
            if word in post and word not in stop:
                return 1.0
        return 0.0

    new_candidates = list()
    for i in candidates:
        if is_valid_content(generate_responses[i], post_sen, stop_words):
            new_candidates.append(i)
    if len(new_candidates) == 0:
        new_candidates = candidates
    return new_candidates


def select_best_response(generate_words, pred_scores, post_data, emotion_labels, emotion_word_dict, batch_size,
                         stop_words, start_id, end_id):
    """
    select best response per batch
    :param generate_words: [batch, beam, max len]
    :param pred_scores: [batch, beam]
    :param post_data: [batch, max len]
    :param emotion_labels: [batch]
    :param emotion_word_dict: {emotion:{word: count}}
    :param batch_size:
    :param stop_words: all the stop words
    :param start_id:<ss>
    :param end_id:<es>
    :return: [batch, max len]
    """
    batch_response_words = []
    for i in range(batch_size):
        this_emotion_words = emotion_word_dict[id2emotion[emotion_labels[i]]].keys()
        emotion_candidates = compute_emotion_candidates(generate_words[i], this_emotion_words)
        content_candidates = compute_content_candidates(generate_words[i], emotion_candidates, post_data[i], stop_words)
        best_index = content_candidates[0]
        this_response_words = generate_words[i, best_index, :]
        batch_response_words.append(this_response_words)
    return batch_response_words


