# -*- coding:utf-8 -*-


model_save_path = "yxl/models"
train_data_path = "yxl/datasets/sentiment_analysis_trainingset.csv"
validate_data_path = "yxl/datasets/sentiment_analysis_validationset.csv"
test_data_path = "yxl/datasets/sentiment_analysis_testa.csv"
test_data_predict_out_path = "yxl/submission"
stopwords_path = "yxl/stopword/stop_words_zh.txt"
sequence_max_len = 300
input_nodes = 300
dictionary_path = "yxl/dictionary"
word_embedding_path = 'yxl/pre-embedding/merge_sgns_bigram_char300.txt'
epochs = 1