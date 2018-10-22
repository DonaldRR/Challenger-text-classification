# -*- coding:utf-8 -*-

import pandas as pd
import jieba
from tqdm import *
import config
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from copy import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Preprocessor():
    def __init__(self, dictionary = None):
        self.tokenizer = Tokenizer()

    def remove_stop_words(self, content, tags):

        f = open(config.stopwords_path)
        stop_words = f.readlines()
        stop_words = [t[:-1] for t in stop_words]

        removed_stop_content = []
        removed_stop_content_tags = []
        for i in tqdm(range(len(content))):
            tmp_content = []
            tmp_content_tags = []
            for j in range(len(content[i])):
                if content[i][j] not in stop_words:
                    tmp_content.append(content[i][j])
                    tmp_content_tags.append(tags[i][j])
            removed_stop_content.append(tmp_content)
            removed_stop_content_tags.append(tmp_content_tags)

        return removed_stop_content, removed_stop_content_tags

    def divide_content_tag(self, content):
        new_content = []
        new_content_tag = []
        for i in tqdm(range(len(content))):
            l = content[i].replace('\n', '')\
                .replace(' ', '')\
                .replace('#', '')\
                .replace('\u3000', '')\
                .split(',')
            new_line = []
            new_tag = []
            for t in l:
                word_tag = t.split('/')
                if len(word_tag) == 2 and word_tag[1].encode('UTF-8').isalpha():
                    new_line.append(word_tag[0])
                    new_tag.append(word_tag[1])
            new_content.append(new_line)
            new_content_tag.append(new_tag)

        return new_content, new_content_tag

    def encode_tag(self, tags):
        # Return unique word tags
        unique_tags = []
        for t in tags:
            unique_tags += t
        unique_tags = list(set(unique_tags))

        # Encoding tags
        enc = LabelEncoder()
        enc.fit(unique_tags)

        return [enc.transform(tags[i]) for i in range(len(tags))]

    def preprocess_content(self, Train_raw):
        #预处理文本文件，分词，去停词，然后返回相应的内容
        #  filepath = config.stopwords_path
        #  stopwords = stopwordslist(filepath)
        content_all = []
        for i in tqdm(range(len(Train_raw))):
            content = Train_raw[i]
            out_str = ''
            content_cut = jieba.cut(content, cut_all = True)
            for word in content_cut:
                if word != '' and '\n':
                    out_str = out_str + ' ' + word
            out_str = out_str + '\n'
            content_all.append(out_str)

        return content_all

    def preprocess_labels(self, label_ori):
        #对label进行预处理，输入是一个label，返回的是onehot编码后的label。返回的可以直接用于训练

        label = label_ori.as_matrix()
        label[label == -1] = 2
        label[label == -2] = 3
        label_onehot = tf.keras.utils.to_categorical(label)
        return label_onehot

    def preprocess_text(self,  content, tags=None, tag_flag=False, train_flag = True):
        #将文本转化为数字，content是分词完了之后的句子，有他的长度
        if train_flag == True:
            #如果是训练的时候，则不会读取,会对文本进行拟合
            print("Constructing Dictionary ...")
            self.tokenizer.fit_on_texts(content)
            Train_sequence = self.tokenizer.texts_to_sequences(content)
            dictionary = self.tokenizer.word_index

            #保存词典
            file = config.dictionary_path + "dictionary.txt"
            f = open(file,'w')
            f.write(str(dictionary))
            f.close()

            sequence_pad = tf.keras.preprocessing.sequence.pad_sequences(Train_sequence, maxlen=config.sequence_max_len,value=0.0, padding = 'pre')
        else:
            #读取训练好的词典
            file = config.dictionary_path + "dictionary.txt"
            fr = open(file,'r')
            dictionary = eval(fr.read())   #读取的str转换为字典
            fr.close()

            #将词典运用在分词中，同时进行分词
            self.tokenizer.word_index = dictionary
            Train_sequence = self.tokenizer.texts_to_sequences(content)

            sequence_pad = tf.keras.preprocessing.sequence.pad_sequences(Train_sequence, maxlen=config.sequence_max_len,value=0.0, padding = 'pre')

        tags_pad = []
        if tag_flag:
            tags_pad = tf.keras.preprocessing.sequence.pad_sequences(tags, maxlen=config.sequence_max_len, value=0., padding='pre')

        return sequence_pad, tags_pad

    def removeData(self, trainData, trainLabel):
        print('Starting removing ......')
        trainData = np.array(trainData)
        trainLabel = np.array(trainLabel)
        print(trainData.shape)
        print(trainLabel.shape)
        num_train = trainData.shape[1]
        rm_Idxs20 = [[3], [3], [3], [3], [1, 3],
					 [3], [3], [3], [3], [3],
					 [1, 3], [3], [3], [1, 3], [1, 3],
					 [0, 1], [1, 3], [3], [1], [1, 3]]

        trans_matrix = []
        for i in range(len(rm_Idxs20)):
            tmp_list = [0] * 4
            for j in range(len(rm_Idxs20[i])):
                tmp_list[rm_Idxs20[i][j]] = 1
            trans_matrix.append(tmp_list)

        Idxs_candidate_to_remove = []

        for i in tqdm(range(105000)):
            if np.sum(np.sum(np.multiply(trainLabel[:, i, :], trans_matrix), axis=0)) == 20:
                Idxs_candidate_to_remove.append(i)

        trainData = np.delete(trainData, Idxs_candidate_to_remove, axis=0)
        trainLabel = np.delete(trainLabel, Idxs_candidate_to_remove, axis=1)
        print('Tuples removed.')
        return trainData, trainLabel

    def shuffle(self, trainData, trainLabel):
        idx_list = [i for i in range(len(trainData))]
        idx_list = np.random.shuffle(idx_list)
        Train_sequence = np.squeeze(trainData[idx_list, :], axis=0)
        Train_label = np.squeeze(trainLabel[:, idx_list], axis=1)
        return Train_sequence, Train_label

    def replicate_data(self, trainData, trainLabel, trainTags=[]):
        trainData = np.array(trainData)
        trainLabel = np.array(trainLabel)
        num_train = trainLabel.shape[1]
        label_distribuition = np.sum(trainLabel, axis=1) / 105000
        num_classes = label_distribuition.shape[0]
        num_subclasses = label_distribuition.shape[1]

        num_groups = len(config.class_group)

        vote_code_count = [[-23, -1, 0, 1, 5],
                           [-31, -1, 0, 1, 5],
                           [-23, -1, 0, 1, 5],
                           [-33, -1, 0, 1, 5],
                           [-23, -1, 0, 1, 5],
                           [-33, -1, 0, 1, 5]]
        vote_code_rate = [0.6, 0.4, 0.15, 0.05, 0.]
        label_votes = np.zeros(shape=(len(vote_code_count), num_classes, num_subclasses), dtype=float)

        # Create voting matrix
        for g in range(len(vote_code_count)):
            for row in range(num_classes):
                for col in range(num_subclasses):
                    label_votes[g][row][col] = vote_code_count[g][5 - np.sum((label_distribuition[row][col] > vote_code_rate), dtype=int)]

        # Count num of replicates
        label_replicate_counts = np.ones(shape=(num_groups, num_train))
        _Train_label = np.vstack(([np.expand_dims(copy(trainLabel), axis=0)] * num_groups))
        replicate_coef = [4, 6, 4, 2, 4, 3]

        print('Compute numbers of replicates...')
        for g in tqdm(range(num_groups)):
            for i in range(num_train):
                tmp_count = np.sum(
                    np.sum(np.multiply(trainLabel[config.class_group[g], i, :], label_votes[g][config.class_group[g], :]), axis=1),
                    axis=0)
                if tmp_count > 0:
                    label_replicate_counts[g][i] = tmp_count * replicate_coef[g]
                    # label_replicate_counts[i] = np.sum(np.sum(np.multiply(Train_label[:,i,:], label_votes), axis=1), axis=0)
                _Train_label[g, :, i, :] *= label_replicate_counts[g][i]

        # Generate replicate data
        replicates_train_datas = []
        replicates_train_labels = []
        replicates_train_tags = []
        for g in range(num_groups):
            replicates_Train_label = copy(trainLabel)[config.class_group[g]]
            replicates_Train_seq = copy(trainData)
            replicates_train_tag = copy(trainTags)
            print("\nReplicating {}'th data set ...".format(g+1))
            for i in tqdm(range(num_train)):
                if label_replicate_counts[g][i] > 1:
                    num_replicate = int(label_replicate_counts[g][i]) - 1
                    replicates_Train_label = np.concatenate((replicates_Train_label,
                                                             [[t] * num_replicate for t in
                                                              trainLabel[config.class_group[g], i, :]]), axis=1)
                    replicates_Train_seq = np.concatenate((replicates_Train_seq, [trainData[i, :]] * num_replicate), axis=0)
                    replicates_train_tag = np.concatenate((replicates_train_tag, [trainTags[i, :]] * num_replicate), axis=0)
            replicates_train_datas.append(replicates_Train_seq)
            replicates_train_labels.append(replicates_Train_label)
            replicates_train_tags.append(replicates_train_tag)

        return replicates_train_datas, replicates_train_labels, replicates_train_tags

