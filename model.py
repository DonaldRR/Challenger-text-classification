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

    def __init__(self, output_nodes,nn_type='lstm', model_path = None):
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
            self.model = self.ModelCreator(config.input_nodes, output_nodes)
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

    def ModelCreator(self, input_node, output_nodes):
        input_nodes = Input(shape = (input_node,))
        a = Embedding(len(self.dictionary)+1,output_dim=300,  # 词向量维度
							weights=[self.embedding_matrix],
							input_length=config.sequence_max_len,  # 文本或者句子截断长度
							trainable=False)(input_nodes)

        filter_fc1 = 64
        filter_fc2 = 32
        if self.nnType is 'lstm':
            lstm = LSTM(256, kernel_regularizer=regularizers.l1(0.01),
						recurrent_regularizer=regularizers.l1(0.01),
						bias_regularizer=regularizers.l1(0.01),
						activity_regularizer=regularizers.l1(0.01))(a)
            pool_ = Lambda(lambda x:expand_dims(x, axis=2))(lstm)
            pool1 = MaxPool1D(pool_size=3)(pool_)
            pool2 = MaxPool1D(pool_size=5)(pool_)
            concat_pool = Concatenate(axis=1)([pool_, pool1, pool2])
            flatten = Flatten()(concat_pool)

            if output_nodes == 2:
                feature11 = Dense(filter_fc1, activation='relu')(flatten)
                feature21 = Dense(filter_fc2, activation='relu')(feature11)
                dp1 = Dropout(0.3)(feature21)
                output1 = Dense(4, activation='softmax')(dp1)
                feature12 = Dense(filter_fc1, activation='relu')(flatten)
                feature22 = Dense(filter_fc2, activation='relu')(feature12)
                dp2 = Dropout(0.3)(feature22)
                output2 = Dense(4, activation='softmax')(dp2)
                output_finals = [output1, output2]
            if output_nodes == 3:
                feature11 = Dense(filter_fc1, activation='relu')(flatten)
                feature21 = Dense(filter_fc2, activation='relu')(feature11)
                dp1 = Dropout(0.3)(feature21)
                output1 = Dense(4, activation='softmax')(dp1)
                feature12 = Dense(filter_fc1, activation='relu')(flatten)
                feature22 = Dense(filter_fc2, activation='relu')(feature12)
                dp2 = Dropout(0.3)(feature22)
                output2 = Dense(4, activation='softmax')(dp2)
                feature13 = Dense(filter_fc1, activation='relu')(flatten)
                feature23 = Dense(filter_fc2, activation='relu')(feature13)
                dp3 = Dropout(0.3)(feature23)
                output3 = Dense(4, activation='softmax')(dp3)
                output_finals = [output1, output2, output3]
            if output_nodes == 2:
                feature11 = Dense(filter_fc1, activation='relu')(flatten)
                feature21 = Dense(filter_fc2, activation='relu')(feature11)
                dp1 = Dropout(0.3)(feature21)
                output1 = Dense(4, activation='softmax')(dp1)
                feature12 = Dense(filter_fc1, activation='relu')(flatten)
                feature22 = Dense(filter_fc2, activation='relu')(feature12)
                dp2 = Dropout(0.3)(feature22)
                output2 = Dense(4, activation='softmax')(dp2)
                feature13 = Dense(filter_fc1, activation='relu')(flatten)
                feature23 = Dense(filter_fc2, activation='relu')(feature13)
                dp3 = Dropout(0.3)(feature23)
                output3 = Dense(4, activation='softmax')(dp3)
                feature14 = Dense(filter_fc1, activation='relu')(flatten)
                feature24 = Dense(filter_fc2, activation='relu')(feature14)
                dp4 = Dropout(0.3)(feature24)
                output4 = Dense(4, activation='softmax')(dp4)
                output_finals = [output1, output2, output3, output4]

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

		

