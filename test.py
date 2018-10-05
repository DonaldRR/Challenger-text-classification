import config
import numpy as np
from model import TextClassifier
from Preprocessing import Preprocessor

Train_seq = np.load(config.train_sequence_path)
Train_label = np.load(config.train_label_path)

p = Preprocessor()
Train_seqs, Train_labels = p.replicate_data(Train_seq, Train_label)