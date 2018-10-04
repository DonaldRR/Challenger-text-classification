import config
import numpy as np

Train_label = np.load(config.train_label_path)
Train_seq = np.load(config.train_sequence_path)

def removeData(trainData, trainLabel):
    num_train = trainData.shape[1]

    rm_Idxs20 = [[3],[3],[3],[3],[1,3],
                 [3],[3],[3],[3],[3],
                 [1,3],[3],[3],[1,3],[1,3],
                 [0,1],[1,3],[3],[1],[1,3]]

    trans_matrix = []
    for i in range(len(rm_Idxs20)):
        tmp_list = [0]*4
        for j in range(len(rm_Idxs20[i])):
            tmp_list[rm_Idxs20[i][j]] = 1
        trans_matrix.append(tmp_list)

    from tqdm import tqdm
    Idxs_candidate_to_remove = []
    for i in tqdm(range(num_train)):
        if np.sum(np.sum(np.multiply(Train_label[:,i,:],trans_matrix), axis=0)) == 20:
            Idxs_candidate_to_remove.append(i)

    trainData = np.delete(trainData, Idxs_candidate_to_remove, axis=1)
    trainLabel = np.delete(trainLabel, Idxs_candidate_to_remove, axis=0)

    return trainData, trainLabel