# -*- coding:utf-8 -*-


import config
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D, Average, ReLU, Add, Reshape, Bidirectional,CuDNNLSTM, Dense, Embedding, Activation, Input, Flatten, BatchNormalization, Conv1D, MaxPooling1D, Concatenate, LSTM, Dropout,MaxPool1D, Conv2D, MaxPooling2D, Lambda, AveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from sklearn import metrics
from tensorflow import expand_dims, squeeze
import tensorflow as tf
from CapsuleLayers import Capsule


class TextClassifier():

    def __init__(self, nn_type='lstm', model_path = None):
        #读取词典
        file = config.dictionary_path + "dictionary.txt"
        fr = open(file,'r')
        self.dictionary = eval(fr.read())   #读取的str转换为字典
        fr.close()

        #读取预训练的wordembedding
        self.embedding_matrix = self.load_embedding_weight()

        self.clusters_meas = np.load('clusters_means.npy')

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

    def ModelCreator(self, input_nodes):
        input_ = Input(shape = (input_nodes,))

        embd_seq = Embedding(len(self.dictionary)+1,output_dim=300,  # 词向量维度
							weights=[self.embedding_matrix],
							input_length=config.sequence_max_len,  # 文本或者句子截断长度
							trainable=False)(input_)

        # embd_seq_1 = Lambda(lambda x:x[:,:198])(embd_seq)
        # embd_seq_2 = Lambda(lambda x: x[:,198:2*198])(embd_seq)
        # embd_seq_3 = Lambda(lambda x: x[:,2*198:])(embd_seq)
        # embd_seq_ = Average()([embd_seq_1, embd_seq_2, embd_seq_3])

        # sdp = SpatialDropout1D(0.3)(embd_seq)

        lstm = Bidirectional(CuDNNLSTM(256))(embd_seq)
        lstm_ = Lambda(lambda x: expand_dims(x, axis=1))(lstm)
        capsules_list = []
        # bn0_1 = BatchNormalization()(lstm)
        # act0_1 = ReLU()(bn0_1)
        # expd = Lambda(lambda x: expand_dims(x, axis=2))(act0_1)
        # conv0_1 = Conv1D(64, 3, 1, 'same')(expd)
        # bn0_2 = BatchNormalization()(conv0_1)
        # act0_2 = ReLU()(bn0_2)
        # conv0_2 = Conv1D(64, 3, 1, 'same')(act0_2)

        #block1
        pool1_list = []
        bn1_1_list = []
        act1_1_list = []
        conv1_1_list = []
        bn1_2_list = []
        act1_2_list = []
        conv1_2_list = []
        add1_list = []

        #block2
        pool2_list = []
        bn2_1_list = []
        act2_1_list = []
        conv2_1_list = []
        bn2_2_list = []
        act2_2_list = []
        conv2_2_list = []
        add2_list = []


        pool7_list = []
        flat_list = []

        fc1_list = []
        fc2_list = []
        dp1_list = []
        dp2_list = []
        outputs_list = []

        for i in range(20):
            capsules_list.append(Capsule(10, 32, 5, True)(lstm_))
            # # block1
            # pool1_list.append(MaxPool1D(2)(conv0_2))
            # # bn1_1_list.append(BatchNormalization()(pool1_list[i]))
            # act1_1_list.append(ReLU()(pool1_list[i]))
            # conv1_1_list.append(Conv1D(64, 3, 1, 'same')(act1_1_list[i]))
            # # bn1_2_list.append(BatchNormalization()(conv1_1_list[i]))
            # act1_2_list.append(ReLU()(conv1_1_list[i]))
            # conv1_2_list.append(Conv1D(64, 3, 1, 'same')(act1_2_list[i]))
            # add1_list.append(Add()([pool1_list[i], conv1_2_list[i]]))
            #
            # # block2
            # pool2_list.append(MaxPool1D(2)(add1_list[i]))
            # # bn2_1_list.append(BatchNormalization()(pool2_list[i]))
            # act2_1_list.append(ReLU()(pool2_list[i]))
            # conv2_1_list.append(Conv1D(64, 3, 1, 'same')(act2_1_list[i]))
            # # bn2_2_list.append(BatchNormalization()(conv2_1_list[i]))
            # act2_2_list.append(ReLU()(conv2_1_list[i]))
            # conv2_2_list.append(Conv1D(64, 3, 1, 'same')(act2_2_list[i]))
            # add2_list.append(Add()([pool2_list[i], conv2_2_list[i]]))
            #
            # pool7_list.append(MaxPool1D(2)(add2_list[i]))
            flat_list.append(Flatten()(capsules_list[i]))

            fc1_list.append(Dense(64, activation='relu')(flat_list[i]))
            dp1_list.append(Dropout(0.4)(fc1_list[i]))
            fc2_list.append(Dense(64, activation='relu')(dp1_list[i]))
            dp2_list.append(Dropout(0.4)(fc2_list[i]))
            outputs_list.append(Dense(4, activation='softmax')(dp2_list[i]))

        model = Model(inputs=input_, outputs=outputs_list, name='base')
        print(model.summary())
        return model

    def train(self, train_data, train_label, val_data = None, val_label = None):
        sgd = SGD(lr=0.004, decay=0.001)
        adam = Adam(lr=0.001, decay=0.05)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(train_data, train_label, epochs=config.epochs, batch_size=64, validation_data=[val_data, val_label])

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

		

