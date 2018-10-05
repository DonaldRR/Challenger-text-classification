# -*- coding:utf-8 -*-

class_group = [[0, 1, 2],
               [3, 4, 5, 6],
               [7, 8, 9],
               [10, 11, 12, 13],
               [14, 15, 16, 17],
               [18, 19]]
model_save_path = "models"
train_data_path = "datasets/sentiment_analysis_trainingset.csv"
validate_data_path = "datasets/sentiment_analysis_validationset.csv"
test_data_path = "datasets/sentiment_analysis_testa.csv"
test_data_predict_out_path = "submission"
stopwords_path = "stop_words_zh.txt"
sequence_max_len = 300
input_nodes = 300
dictionary_path = ""
word_embedding_path = '/home/donald/datasets/ChallengerText_DS/wordVec'
epochs = 5
train_sequence_path = 'preprocessed_data/train_seq.npy'
validation_sequence_path = 'preprocessed_data/validation_seq.npy'
test_sequence_path = 'preprocessed_data/test_seq.npy'
train_label_path = 'preprocessed_data/train_label.npy'
validation_label_path = 'preprocessed_data/validation_label.npy'
f1_path = 'f1/'