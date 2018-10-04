from copy import copy
import config
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

Train_label = np.load(config.train_label_path)
Train_seq = np.load(config.train_sequence_path)

num_train = Train_label.shape[1]

label_distribuition = np.sum(Train_label, axis=1)/105000
num_classes = label_distribuition.shape[0]
num_subclasses = label_distribuition.shape[1]

# Compute voting for each subclasses
vote_code_count = [-4, -3, -2, 0, 1, 2, 4]
vote_code_rate = [0.7, 0.5, 0.35, 0.15, 0.08, 0.04, 0.]
label_votes = np.zeros(label_distribuition.shape, dtype=float)

for row in range(num_classes):
    for col in range(num_subclasses):
        label_votes[row][col] = vote_code_count[6 - np.sum((label_distribuition[row][col] > vote_code_rate), dtype=int)]
# print(label_votes)

# Compute replicates for each sample
label_replicate_counts = np.zeros(num_train)

for i in tqdm(range(num_train)):
    label_replicate_counts[i] = np.sum(np.sum(np.multiply(Train_label[:,i,:], label_votes), axis=1), axis=0)

print('Num of replicates:{}'.format(label_replicate_counts[np.nonzero(np.array((label_replicate_counts>0), dtype=int))]))

# Replicate samples
print('Start Replicate ...')
replicates_Train_label = copy(Train_label)
replicates_Train_seq = copy(Train_seq)

for i in tqdm(range(num_train)):
    num_replicate = int(label_replicate_counts[i])
    if num_replicate > 0:
        replicates_Train_label = np.concatenate((replicates_Train_label,
                                                 [[t]*num_replicate for t in Train_label[:,i,:]]), axis=1)
        replicates_Train_seq = np.concatenate((replicates_Train_seq, [Train_seq[i,:]] * num_replicate),axis=0)
print('Replicate Finished.')