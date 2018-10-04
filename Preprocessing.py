# -*- coding:utf-8 -*-

import pandas as pd
import jieba
from tqdm import *
import config
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np

class Preprocessor():
	def __init__(self, dictionary = None):
		self.tokenizer = Tokenizer()


	def preprocess_content(self, Train_raw):
		#预处理文本文件，分词，去停词，然后返回相应的内容
		# filepath = config.stopwords_path
		# stopwords = stopwordslist(filepath)
		
		content_all = []
		for i in tqdm(range(len(Train_raw))):
			content = Train_raw[i]
			out_str = ''
			content_cut = jieba.cut(content, cut_all = False)
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


	def preprocess_text(self,  content, train_flag = True):
		#将文本转化为数字，content是分词完了之后的句子，有他的长度
		if train_flag == True:

			#如果是训练的时候，则不会读取,会对文本进行拟合
			self.tokenizer.fit_on_texts(content)
			Train_sequence = self.tokenizer.texts_to_sequences(content)
			print("序列化完成，正在保存词典")
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

		return sequence_pad

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