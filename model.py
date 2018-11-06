# -*- coding:utf-8 -*-


import config
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, CuDNNGRU, multiply, Permute, RepeatVector, BatchNormalization, \
    SpatialDropout1D, Average, ReLU, Add, Reshape, Bidirectional,CuDNNLSTM, Dense, Embedding, Activation, \
    Input, Flatten, BatchNormalization, Conv1D, MaxPooling1D, Concatenate, LSTM, Dropout,MaxPool1D, Conv2D,\
    MaxPooling2D, Lambda, AveragePooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from sklearn import metrics
from tensorflow import expand_dims, squeeze
import tensorflow as tf
from CapsuleLayers import Capsule
import tensorflow.keras.backend as K
from tqdm import tqdm
from keras import backend as K
from utils import CustomModelCheckPoint



def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(y_true * y_pred, axis=0)
        # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(y_true, axis=0)
        # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(y_true * y_pred, axis=0)
        # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(y_pred, axis=0)
        # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    y_pred = tf.one_hot(tf.argmax(y_pred, axis=1), 4)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))

class TextClassifier():

    def __init__(self, nn_type='lstm', embedding_weights=False, sample_weights=None ,model_path = None):
        #读取词典
        file = config.dictionary_path + "dictionary.txt"
        fr = open(file,'r')
        self.dictionary = eval(fr.read())   #读取的str转换为字典
        fr.close()

        #读取预训练的wordembedding
        if not embedding_weights:
            self.embedding_matrix = self.load_embedding_weight()
            np.save(config.embedding_weights_path, self.embedding_matrix)
        else:
            self.embedding_matrix = np.load(config.embedding_weights_path)

        self.clusters_meas = np.load('clusters_means.npy')

        #初始化模型
        self.sample_weights = sample_weights

        self.nnType = nn_type
        if model_path == None:
            self.model = self.ModelCreator(config.input_nodes)
        else:
            self.model = load_model(model_path)

    def load_embedding_weight(self):


        f=open(config.word_embedding_path,"r",encoding="utf-8")
        ## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
        l,d=np.array(f.readline().split(), dtype=np.int)
        ## 创建词向量索引字典
        embeddings_index={}
        for line in f:
            ## 读取词向量文件中的每一行
            values=line.split()[:d+1]
            ## 获取当前行的词
            word=values[0]
            ## 获取当前词的词向量
            try:
                coefs=np.array(values[1:],dtype="float32")
            except:
                continue
            ## 将读入的这行词向量加入词向量索引字典
            embeddings_index[word]=coefs
        f.close()

        num_words = len(self.dictionary) + 1  # 词汇表数量
        embedding_matrix = np.zeros((num_words, 300))  # 20000*300
        for word, i in tqdm(self.dictionary.items()):
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
							trainable=False,
                             mask_zero=False)(input_)

        gru = Bidirectional(CuDNNGRU(256))(embd_seq)
        # gru = Lambda(lambda x: expand_dims(x, axis=1))(gru)

        outputs_list= []
        for i in range(20):
            # cap = Capsule(10,16,5,True)(gru)
            # cap = Flatten()(cap)
            out = Dense(64)(gru)
            out = Dropout(0.5)(out)
            out = Dense(4, activation='softmax')(out)
            outputs_list.append(out)

        model = Model(inputs=input_, outputs=outputs_list, name='base')
        print(model.summary())
        return model

    def train(self, train_data, train_label, val_data=None, val_label=None):

        # sgd = SGD(lr=0.00005, decay=0.001)
        adam = Adam(lr=0.0001)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1, 'categorical_accuracy'])
        hist = self.model.fit(train_data, train_label, epochs=config.epochs, batch_size=64,
                              validation_data=[val_data, val_label], sample_weight = self.sample_weights)

        return hist

    def save(self, model_name):
        save_path = config.model_save_path + "/" + model_name
        self.model.save(save_path)

    def predict(self, test_data):
        y_pred = self.model.predict(test_data)
        return np.argmax(y_pred, axis=0)

    def evaluate(self, model, data, true_label, num_labels=20):
        data_pred = model.predict(data)
        y_pred = []
        y_true = []

        for i in range(num_labels):
            y_pred.append(np.argmax(data_pred[i], axis=1))
            y_true.append(np.argmax(true_label[i], axis=1))

        f1 = []
        for j in range(len(y_pred)):
            f1.append(metrics.f1_score(y_true[j], y_pred[j], average = 'macro'))
        return f1

    def _f1_monitor(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)

        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        return f1