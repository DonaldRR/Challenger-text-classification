# -*- coding:utf-8 -*-


import config
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Activation, Input, Flatten, BatchNormalization, Conv1D, MaxPooling1D, Concatenate, LSTM, Dropout,MaxPool1D, Conv2D, MaxPooling2D, Lambda, AveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import numpy as np
from sklearn import metrics
from tensorflow import expand_dims


class TextClassifier():

	def __init__(self, nn_type='lstm', model_path = None):
		#读取词典
		file = config.dictionary_path + "/dictionary.txt"
		fr = open(file,'r')
		self.dictionary = eval(fr.read())   #读取的str转换为字典
		fr.close()

		#读取预训练的wordembedding
		self.embedding_matrix = self.load_embedding_weight()

		#初始化模型
		self.nnType = nn_type
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
		input_nodes = Input(shape = (input_node,))
		a = Embedding(len(self.dictionary)+1,output_dim=300,  # 词向量维度
							weights=[self.embedding_matrix],
							input_length=config.sequence_max_len,  # 文本或者句子截断长度
							trainable=False)(input_nodes)

		if self.nnType is 'lstm':
			filter_fc1 = 64
			lstm = LSTM(256, kernel_regularizer=regularizers.l1(0.01),
						recurrent_regularizer=regularizers.l1(0.01),
						bias_regularizer=regularizers.l1(0.01),
						activity_regularizer=regularizers.l1(0.01))(a)
			pool_ = Lambda(lambda x:expand_dims(x, axis=2))(lstm)
			pool1 = MaxPool1D(pool_size=3)(pool_)
			pool2 = MaxPool1D(pool_size=5)(pool_)
			pool3 = MaxPool1D(pool_size=7)(pool_)
			concat_pool = Concatenate(axis=1)([pool_, pool1, pool2, pool3])
			flatten = Flatten()(concat_pool)

			feature_11 = Dense(filter_fc1, activation='relu')(flatten)
			feature_12 = Dense(filter_fc1, activation='relu')(flatten)
			feature_13 = Dense(filter_fc1, activation='relu')(flatten)
			feature_14 = Dense(filter_fc1, activation='relu')(flatten)
			feature_15 = Dense(filter_fc1, activation='relu')(flatten)
			feature_16 = Dense(filter_fc1, activation='relu')(flatten)
			feature_17 = Dense(filter_fc1, activation='relu')(flatten)
			feature_18 = Dense(filter_fc1, activation='relu')(flatten)
			feature_19 = Dense(filter_fc1, activation='relu')(flatten)
			feature_110 = Dense(filter_fc1, activation='relu')(flatten)
			feature_111 = Dense(filter_fc1, activation='relu')(flatten)
			feature_112 = Dense(filter_fc1, activation='relu')(flatten)
			feature_113 = Dense(filter_fc1, activation='relu')(flatten)
			feature_114 = Dense(filter_fc1, activation='relu')(flatten)
			feature_115 = Dense(filter_fc1, activation='relu')(flatten)
			feature_116 = Dense(filter_fc1, activation='relu')(flatten)
			feature_117 = Dense(filter_fc1, activation='relu')(flatten)
			feature_118 = Dense(filter_fc1, activation='relu')(flatten)
			feature_119 = Dense(filter_fc1, activation='relu')(flatten)
			feature_120 = Dense(filter_fc1, activation='relu')(flatten)

		elif self.nnType is 'cnn':
			new_a = Lambda(lambda x: expand_dims(x, axis=3))(a)
			# new_a = expand_dims(a, 3)
			conv1 = Conv2D(filters=64, kernel_size=(3,51), padding='valid')(new_a)
			conv2 = Conv2D(filters=32, kernel_size=(3,51), padding='valid')(conv1)
			# conv3 = Conv2D(filters=32, kernel_size=(5,51), padding='valid')(conv2)
			# conv4 = Conv2D(filters=16, kernel_size=(7,51), padding='valid')(conv3)
			pool1 = AveragePooling2D(pool_size=(4,4))(conv2)
			pool2 = MaxPooling2D(pool_size=(4,4))(pool1)
			flatten = Flatten()(pool2)

			feature_11 = Dense(64, activation='relu')(flatten)
			feature_12 = Dense(64, activation='relu')(flatten)
			feature_13 = Dense(64, activation='relu')(flatten)
			feature_14 = Dense(64, activation='relu')(flatten)
			feature_15 = Dense(64, activation='relu')(flatten)
			feature_16 = Dense(64, activation='relu')(flatten)
			feature_17 = Dense(64, activation='relu')(flatten)
			feature_18 = Dense(64, activation='relu')(flatten)
			feature_19 = Dense(64, activation='relu')(flatten)
			feature_110 = Dense(64, activation='relu')(flatten)
			feature_111 = Dense(64, activation='relu')(flatten)
			feature_112 = Dense(64, activation='relu')(flatten)
			feature_113 = Dense(64, activation='relu')(flatten)
			feature_114 = Dense(64, activation='relu')(flatten)
			feature_115 = Dense(64, activation='relu')(flatten)
			feature_116 = Dense(64, activation='relu')(flatten)
			feature_117 = Dense(64, activation='relu')(flatten)
			feature_118 = Dense(64, activation='relu')(flatten)
			feature_119 = Dense(64, activation='relu')(flatten)
			feature_120 = Dense(64, activation='relu')(flatten)

		filter_fc2 = 32
		feature_21 = Dense(filter_fc2, activation='relu')(feature_11)
		feature_22 = Dense(filter_fc2, activation='relu')(feature_12)
		feature_23 = Dense(filter_fc2, activation='relu')(feature_13)
		feature_24 = Dense(filter_fc2, activation='relu')(feature_14)
		feature_25 = Dense(filter_fc2, activation='relu')(feature_15)
		feature_26 = Dense(filter_fc2, activation='relu')(feature_16)
		feature_27 = Dense(filter_fc2, activation='relu')(feature_17)
		feature_28 = Dense(filter_fc2, activation='relu')(feature_18)
		feature_29 = Dense(filter_fc2, activation='relu')(feature_19)
		feature_210 = Dense(filter_fc2, activation='relu')(feature_110)
		feature_211 = Dense(filter_fc2, activation='relu')(feature_111)
		feature_212 = Dense(filter_fc2, activation='relu')(feature_112)
		feature_213 = Dense(filter_fc2, activation='relu')(feature_113)
		feature_214 = Dense(filter_fc2, activation='relu')(feature_114)
		feature_215 = Dense(filter_fc2, activation='relu')(feature_115)
		feature_216 = Dense(filter_fc2, activation='relu')(feature_116)
		feature_217 = Dense(filter_fc2, activation='relu')(feature_117)
		feature_218 = Dense(filter_fc2, activation='relu')(feature_118)
		feature_219 = Dense(filter_fc2, activation='relu')(feature_119)
		feature_220 = Dense(filter_fc2, activation='relu')(feature_120)

		drop_rate = 0.4
		dp_1 = Dropout(drop_rate)(feature_21)
		dp_2 = Dropout(drop_rate)(feature_22)
		dp_3 = Dropout(drop_rate)(feature_23)
		dp_4 = Dropout(drop_rate)(feature_24)
		dp_5 = Dropout(drop_rate)(feature_25)
		dp_6 = Dropout(drop_rate)(feature_26)
		dp_7 = Dropout(drop_rate)(feature_27)
		dp_8 = Dropout(drop_rate)(feature_28)
		dp_9 = Dropout(drop_rate)(feature_29)
		dp_10 = Dropout(drop_rate)(feature_210)
		dp_11 = Dropout(drop_rate)(feature_211)
		dp_12 = Dropout(drop_rate)(feature_212)
		dp_13 = Dropout(drop_rate)(feature_213)
		dp_14 = Dropout(drop_rate)(feature_214)
		dp_15 = Dropout(drop_rate)(feature_215)
		dp_16 = Dropout(drop_rate)(feature_216)
		dp_17 = Dropout(drop_rate)(feature_217)
		dp_18 = Dropout(drop_rate)(feature_218)
		dp_19 = Dropout(drop_rate)(feature_219)
		dp_20 = Dropout(drop_rate)(feature_220)

		output_1 = Dense(4, activation='softmax')(dp_1)
		output_2 = Dense(4, activation='softmax')(dp_2)
		output_3 = Dense(4, activation='softmax')(dp_3)
		output_4 = Dense(4, activation='softmax')(dp_4)
		output_5 = Dense(4, activation='softmax')(dp_5)
		output_6 = Dense(4, activation='softmax')(dp_6)
		output_7 = Dense(4, activation='softmax')(dp_7)
		output_8 = Dense(4, activation='softmax')(dp_8)
		output_9 = Dense(4, activation='softmax')(dp_9)
		output_10 = Dense(4, activation='softmax')(dp_10)
		output_11 = Dense(4, activation='softmax')(dp_11)
		output_12 = Dense(4, activation='softmax')(dp_12)
		output_13 = Dense(4, activation='softmax')(dp_13)
		output_14 = Dense(4, activation='softmax')(dp_14)
		output_15 = Dense(4, activation='softmax')(dp_15)
		output_16 = Dense(4, activation='softmax')(dp_16)
		output_17 = Dense(4, activation='softmax')(dp_17)
		output_18 = Dense(4, activation='softmax')(dp_18)
		output_19 = Dense(4, activation='softmax')(dp_19)
		output_20 = Dense(4, activation='softmax')(dp_20)

		output_finals = [output_1,output_2,output_3,output_4,output_5,output_6,
					output_7,output_8,output_9,output_10,output_11,output_12,
					output_13,output_14,output_15,output_16,output_17,output_18,
					output_19,output_20]
		
		model = Model(inputs=input_nodes, outputs=output_finals, name='base')
		print(model.summary())
		return model

	def train(self, train_data, train_label, val_data = None, val_label = None):
		self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['categorical_accuracy'])
		self.model.fit(train_data, train_label, epochs=config.epochs, batch_size = 64, validation_data = [val_data, val_label])

	def save(self, model_name):
		save_path = config.model_save_path + "/" + model_name
		self.model.save(save_path)

	def predict(self, test_data):
		y_pred = self.model.predict(test_data)
		return np.argmax(y_pred, axis = 0)

	def evaluate(self, model, data, true_label, num_labels=20):

		data_pred = model.predict(data)
		y_pred = []
		y_true = []

		for i in range(num_labels):
			y_pred.append(np.argmax(data_pred[i], axis=1))
			y_true.append(np.argmax(true_label[i], axis=1))

		f1 = []
		for j in range(len(y_pred)):
			f1.append(metrics.f1_score(y_true[j], y_pred[j], average = 'micro'))
		return f1




		

