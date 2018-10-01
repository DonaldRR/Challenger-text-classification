# -*- coding:utf-8 -*-


import config
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Activation, Input, Flatten, BatchNormalization, Conv1D, MaxPooling1D, Concatenate, LSTM, Dropout, Conv2D, MaxPooling2D, Lambda, AveragePooling2D
from tensorflow.keras.models import load_model
import numpy as np
from sklearn import metrics
from tensorflow import expand_dims


class TextClassifier():

	def __init__(self, nn_type='lstm', model_path = None):
		#读取词典
		file = config.dictionary_path + "dictionary.txt"
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
			feature_1 = LSTM(512)(a)

			feature_11 = Dense(64, activation='relu')(feature_1)
			feature_12 = Dense(64, activation='relu')(feature_1)
			feature_13 = Dense(64, activation='relu')(feature_1)
			feature_14 = Dense(64, activation='relu')(feature_1)
			feature_15 = Dense(64, activation='relu')(feature_1)
			feature_16 = Dense(64, activation='relu')(feature_1)
			feature_17 = Dense(64, activation='relu')(feature_1)
			feature_18 = Dense(64, activation='relu')(feature_1)
			feature_19 = Dense(64, activation='relu')(feature_1)
			feature_110 = Dense(64, activation='relu')(feature_1)
			feature_111 = Dense(64, activation='relu')(feature_1)
			feature_112 = Dense(64, activation='relu')(feature_1)
			feature_113 = Dense(64, activation='relu')(feature_1)
			feature_114 = Dense(64, activation='relu')(feature_1)
			feature_115 = Dense(64, activation='relu')(feature_1)
			feature_116 = Dense(64, activation='relu')(feature_1)
			feature_117 = Dense(64, activation='relu')(feature_1)
			feature_118 = Dense(64, activation='relu')(feature_1)
			feature_119 = Dense(64, activation='relu')(feature_1)
			feature_120 = Dense(64, activation='relu')(feature_1)

			feature_21 = Dense(32, activation='relu')(feature_11)
			feature_22 = Dense(32, activation='relu')(feature_12)
			feature_23 = Dense(32, activation='relu')(feature_13)
			feature_24 = Dense(32, activation='relu')(feature_14)
			feature_25 = Dense(32, activation='relu')(feature_15)
			feature_26 = Dense(32, activation='relu')(feature_16)
			feature_27 = Dense(32, activation='relu')(feature_17)
			feature_28 = Dense(32, activation='relu')(feature_18)
			feature_29 = Dense(32, activation='relu')(feature_19)
			feature_210 = Dense(32, activation='relu')(feature_110)
			feature_211 = Dense(32, activation='relu')(feature_111)
			feature_212 = Dense(32, activation='relu')(feature_112)
			feature_213 = Dense(32, activation='relu')(feature_113)
			feature_214 = Dense(32, activation='relu')(feature_114)
			feature_215 = Dense(32, activation='relu')(feature_115)
			feature_216 = Dense(32, activation='relu')(feature_116)
			feature_217 = Dense(32, activation='relu')(feature_117)
			feature_218 = Dense(32, activation='relu')(feature_118)
			feature_219 = Dense(32, activation='relu')(feature_119)
			feature_220 = Dense(32, activation='relu')(feature_120)

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

			feature_21 = Dense(32, activation='relu')(flatten)
			feature_22 = Dense(32, activation='relu')(flatten)
			feature_23 = Dense(32, activation='relu')(flatten)
			feature_24 = Dense(32, activation='relu')(flatten)
			feature_25 = Dense(32, activation='relu')(flatten)
			feature_26 = Dense(32, activation='relu')(flatten)
			feature_27 = Dense(32, activation='relu')(flatten)
			feature_28 = Dense(32, activation='relu')(flatten)
			feature_29 = Dense(32, activation='relu')(flatten)
			feature_210 = Dense(32, activation='relu')(flatten)
			feature_211 = Dense(32, activation='relu')(flatten)
			feature_212 = Dense(32, activation='relu')(flatten)
			feature_213 = Dense(32, activation='relu')(flatten)
			feature_214 = Dense(32, activation='relu')(flatten)
			feature_215 = Dense(32, activation='relu')(flatten)
			feature_216 = Dense(32, activation='relu')(flatten)
			feature_217 = Dense(32, activation='relu')(flatten)
			feature_218 = Dense(32, activation='relu')(flatten)
			feature_219 = Dense(32, activation='relu')(flatten)
			feature_220 = Dense(32, activation='relu')(flatten)

		feature_31 = Dense(16, activation='relu')(feature_21)
		feature_32 = Dense(16, activation='relu')(feature_22)
		feature_33 = Dense(16, activation='relu')(feature_23)
		feature_34 = Dense(16, activation='relu')(feature_24)
		feature_35 = Dense(16, activation='relu')(feature_25)
		feature_36 = Dense(16, activation='relu')(feature_26)
		feature_37 = Dense(16, activation='relu')(feature_27)
		feature_38 = Dense(16, activation='relu')(feature_28)
		feature_39 = Dense(16, activation='relu')(feature_29)
		feature_310 = Dense(16, activation='relu')(feature_210)
		feature_311 = Dense(16, activation='relu')(feature_211)
		feature_312 = Dense(16, activation='relu')(feature_212)
		feature_313 = Dense(16, activation='relu')(feature_213)
		feature_314 = Dense(16, activation='relu')(feature_214)
		feature_315 = Dense(16, activation='relu')(feature_215)
		feature_316 = Dense(16, activation='relu')(feature_216)
		feature_317 = Dense(16, activation='relu')(feature_217)
		feature_318 = Dense(16, activation='relu')(feature_218)
		feature_319 = Dense(16, activation='relu')(feature_219)
		feature_320 = Dense(16, activation='relu')(feature_220)

		dp_1 = Dropout(0.4)(feature_31)
		dp_2 = Dropout(0.4)(feature_32)
		dp_3 = Dropout(0.4)(feature_33)
		dp_4 = Dropout(0.4)(feature_34)
		dp_5 = Dropout(0.4)(feature_35)
		dp_6 = Dropout(0.4)(feature_36)
		dp_7 = Dropout(0.4)(feature_37)
		dp_8 = Dropout(0.4)(feature_38)
		dp_9 = Dropout(0.4)(feature_39)
		dp_10 = Dropout(0.4)(feature_310)
		dp_11 = Dropout(0.4)(feature_311)
		dp_12 = Dropout(0.4)(feature_312)
		dp_13 = Dropout(0.4)(feature_313)
		dp_14 = Dropout(0.4)(feature_314)
		dp_15 = Dropout(0.4)(feature_315)
		dp_16 = Dropout(0.4)(feature_316)
		dp_17 = Dropout(0.4)(feature_317)
		dp_18 = Dropout(0.4)(feature_318)
		dp_19 = Dropout(0.4)(feature_319)
		dp_20 = Dropout(0.4)(feature_320)

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

	def evaluate(self, y_pred, y_true):
		f1 = []
		for j in range(len(y_pred)):
			f1.append(metrics.f1_score(y_true[j], y_pred[j], average = 'micro'))
		return f1




		

