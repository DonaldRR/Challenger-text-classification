# -*- coding:utf-8 -*-


import config
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Activation, Input, Flatten, BatchNormalization, Conv1D, MaxPooling1D, Concatenate, LSTM, Dropout
from tensorflow.keras.models import load_model
import numpy as np
from sklearn import metrics


class TextClassifier():

	def __init__(self, model_path = None):
		#读取词典
		file = config.dictionary_path + "dictionary.txt"
		fr = open(file,'r')
		self.dictionary = eval(fr.read())   #读取的str转换为字典
		fr.close()

		#读取预训练的wordembedding
		self.embedding_matrix = self.load_embedding_weight()

		#初始化模型
		if model_path == None:
			self.model = self.ModelCreator(config.input_nodes)
		else:
			self.model = load_model(model_path)


	def load_embedding_weight(self):

		f=open(config.word_embedding_path,"r",encoding="utf-8")
		## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
		l,w=f.readline().split()
		## 创建词向量索引字典
		embeddings_index={}
		for line in f:
			## 读取词向量文件中的每一行
			values=line.split()
			## 获取当前行的词
			word=values[0]
			## 获取当前词的词向量
			try:
				coefs=np.asarray(values[1:],dtype="float32")
			except:
				continue
			## 将读入的这行词向量加入词向量索引字典
			embeddings_index[word]=coefs
		f.close()

		num_words = len(self.dictionary) + 1  # 词汇表数量
		embedding_matrix = np.zeros((num_words, 300))  # 20000*100
		
		for word, i in self.dictionary.items():
			if i >= num_words:  # 过滤掉根据频数排序后排20000以后的词
				continue
			embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
				
		return embedding_matrix


	
	def ModelCreator(self, input_node):
		input_node = Input(shape = (input_node,))
		a = Embedding(len(self.dictionary)+1,output_dim=300,  # 词向量维度
							weights=[self.embedding_matrix],
							input_length=config.sequence_max_len,  # 文本或者句子截断长度
							trainable=False)(input_node)
		
		feature_1 = LSTM(64)(a)
		feature_2 = LSTM(64)(a)
		feature_3 = LSTM(64)(a)
		feature_4 = LSTM(64)(a)
		feature_5 = LSTM(64)(a)
		feature_6 = LSTM(64)(a)
# 		feature_7 = LSTM(64)(a)
# 		feature_8 = LSTM(64)(a)
# 		feature_9 = LSTM(64)(a)
# 		feature_10 = LSTM(64)(a)
# 		feature_11 = LSTM(64)(a)
# 		feature_12 = LSTM(64)(a)
# 		feature_13 = LSTM(64)(a)
# 		feature_14 = LSTM(64)(a)
# 		feature_15 = LSTM(64)(a)
# 		feature_16 = LSTM(64)(a)
# 		feature_17 = LSTM(64)(a)
# 		feature_18 = LSTM(64)(a)
# 		feature_19 = LSTM(64)(a)
# 		feature_20 = LSTM(64)(a)
		
		
		feature_11 = Dense(32, activation='relu')(feature_1)
		feature_21 = Dense(32, activation='relu')(feature_2)
		feature_31 = Dense(32, activation='relu')(feature_3)
		feature_41 = Dense(32, activation='relu')(feature_4)
		feature_51 = Dense(32, activation='relu')(feature_5)
		feature_61 = Dense(32, activation='relu')(feature_6)
		feature_12 = Dense(32, activation='relu')(feature_11)
		feature_22= Dense(32, activation='relu')(feature_21)
		feature_32 = Dense(32, activation='relu')(feature_31)
		feature_42 = Dense(32, activation='relu')(feature_41)
		feature_52 = Dense(32, activation='relu')(feature_51)
		feature_62 = Dense(32, activation='relu')(feature_61)
		dp_1 = Dropout(0.3)(feature_12)
		dp_2 = Dropout(0.3)(feature_22)
		dp_3 = Dropout(0.3)(feature_32)
		dp_4 = Dropout(0.3)(feature_42)
		dp_5 = Dropout(0.3)(feature_52)
		dp_6 = Dropout(0.3)(feature_62)
# 		feature_71 = Dense(128, activation='relu')(feature_2)
# 		feature_81 = Dense(128, activation='relu')(feature_3)
# 		feature_91 = Dense(128, activation='relu')(feature_3)
# 		feature_101 = Dense(128, activation='relu')(feature_3)
# 		feature_111 = Dense(128, activation='relu')(feature_4)
# 		feature_121 = Dense(128, activation='relu')(feature_4)
# 		feature_131 = Dense(128, activation='relu')(feature_4)
# 		feature_141 = Dense(128, activation='relu')(feature_4)
# 		feature_151 = Dense(128, activation='relu')(feature_5)
# 		feature_161 = Dense(128, activation='relu')(feature_5)
# 		feature_171 = Dense(128, activation='relu')(feature_5)
# 		feature_181 = Dense(128, activation='relu')(feature_5)
# 		feature_191 = Dense(128, activation='relu')(feature_6)
# 		feature_201 = Dense(128, activation='relu')(feature_6)
		

		output_1 = Dense(4, activation='softmax')(dp_1)
		output_2 = Dense(4, activation='softmax')(dp_1)
		output_3 = Dense(4, activation='softmax')(dp_1)
		output_4 = Dense(4, activation='softmax')(dp_2)
		output_5 = Dense(4, activation='softmax')(dp_2)
		output_6 = Dense(4, activation='softmax')(dp_2)
		output_7 = Dense(4, activation='softmax')(dp_2)
		output_8 = Dense(4, activation='softmax')(dp_3)
		output_9 = Dense(4, activation='softmax')(dp_3)
		output_10 = Dense(4, activation='softmax')(dp_3)
		output_11 = Dense(4, activation='softmax')(dp_4)
		output_12 = Dense(4, activation='softmax')(dp_4)
		output_13 = Dense(4, activation='softmax')(dp_4)
		output_14 = Dense(4, activation='softmax')(dp_4)
		output_15 = Dense(4, activation='softmax')(dp_5)
		output_16 = Dense(4, activation='softmax')(dp_5)
		output_17 = Dense(4, activation='softmax')(dp_5)
		output_18 = Dense(4, activation='softmax')(dp_5)
		output_19 = Dense(4, activation='softmax')(dp_6)
		output_20 = Dense(4, activation='softmax')(dp_6)

		output_final = [output_1,output_2,output_3,output_4,output_5,output_6,
					output_7,output_8,output_9,output_10,output_11,output_12,
					output_13,output_14,output_15,output_16,output_17,output_18,
					output_19,output_20]
		
		model = Model(inputs=input_node, outputs=output_final, name='base')
		print(model.summary())
		return model

	def train(self, train_data, train_label, val_data = None, val_label = None):
		self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['categorical_accuracy'])
		self.model.fit(train_data, train_label, epochs=config.epochs, batch_size = 256, validation_data = [val_data, val_label])

	def save(self, model_name):
		save_path = config.model_save_path + "/" + model_name
		self.model.save(save_path)

	def predict(self, test_data):
		y_pred = self.model.predict(test_data)
		return np.argmax(y_pred, axis = 0)

	def evaluate(self, y_pred, y_true):
		f1 = []
		for j in range(len(y_pred)):
			f1.append(metrics.f1_score(y_true[j], y_pred[j], average = 'micro'))
		return f1




		

