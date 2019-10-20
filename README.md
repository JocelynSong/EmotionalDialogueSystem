# EmotionalDialogueSystem
This is the code for paper ["Generating Responses with a Specific Emotion in Dialog"](https://www.aclweb.org/anthology/P19-1359.pdf) in acl 2019.

This code is written in tensorflow, the start file is exp/train_batch.py. You can use the following commands to train this model:
python train_batch.py \
--config_file configguration_file \
--pre_train_word_count_file emotional_word_count_file_in_supervised_emotional_data_with_"word|||count" \
--emotion_words_dir this_is_the_emotional_words_directory_with_six_emotional_word_files \
--post_file post_file_name \
--response_file response_file_name \
--emotion_label_file emotion_label_file_for response \
--embedding_file word_embedding_file \
--train_word_count total_word_count_file_in_stc_data_with_"word|||count" \
--checkpoint_dir saving_dir
