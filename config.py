# -*- coding:utf-8 -*-

class_group = [[0, 1, 2],
               [3, 4, 5, 6],
               [7, 8, 9],
               [10, 11, 12, 13],
               [14, 15, 16, 17],
               [18, 19]]
# Models
model_save_path = "models"
# Data
train_data_path = "datasets/sentiment_analysis_trainingset.csv"
validate_data_path = "datasets/sentiment_analysis_validationset.csv"
test_data_path = "datasets/sentiment_analysis_testa.csv"
ansj_train_data_path = "datasets/ansj_sentiment_trainingset.csv"
ansj_validation_data_path = "datasets/ansj_sentiment_validationset.csv"
ansj_test_data_path = "datasets/ansj_sentiment_testset.csv"
# Submission
test_data_predict_out_path = "submission"
# Stop words
stopwords_path = "stop_words_zh.txt"
# Parameters
sequence_max_len = 400
input_nodes = 400
epochs = 1
# Dictionary
dictionary_path = ""
# Embedding
word_embedding_path = '/home/donald/datasets/ChallengerText_DS/wordVec'
# Preprocessed Data
train_sequence_path = 'preprocessed_data/train_seq.npy'
validation_sequence_path = 'preprocessed_data/validation_seq.npy'
test_sequence_path = 'preprocessed_data/test_seq.npy'
train_label_path = 'preprocessed_data/train_label.npy'
validation_label_path = 'preprocessed_data/validation_label.npy'
train_tags_path = 'preprocessed_data/train_tags.npy'
validation_tags_path = 'preprocessed_data/validation_tags.npy'
test_tags_path = 'preprocessed_data/test_tags.npy'
# F1
f1_path = 'f1/'