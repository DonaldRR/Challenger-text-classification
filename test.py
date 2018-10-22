import config
import numpy as np
from sklearn import metrics

# 读入预处理好的数据
Train_seq_set = np.load(config.train_sequence_path)
Train_tags_set = np.load(config.train_tags_path)
Train_label_set = np.load(config.train_label_path)
Validation_seq = np.load(config.validation_sequence_path)
Validation_tags_set = np.load(config.validation_tags_path)
Validation_label = np.load(config.validation_label_path)

train_input = np.concatenate((Train_seq_set, Train_tags_set), axis=1)
validation_input = np.concatenate((Validation_seq,Validation_tags_set), axis=1)

from model import TextClassifier
clf = TextClassifier()