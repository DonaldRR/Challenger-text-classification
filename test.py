import config
import numpy as np
from model import TextClassifier

Train_seq = open(config.train_sequence_path)
Train_label = open(config.train_label_path)
Validation_seq = open(config.validation_sequence_path)
Validation_label = open(config.validation_label_path)
Test_seq = open(config.test_sequence_path)

clf = TextClassifier()
clf.train(Train_seq, Train_label, Validation_seq, Validation_label)
