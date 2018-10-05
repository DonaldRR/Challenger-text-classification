from Preprocessing import Preprocessor
from model import TextClassifier
import config
import numpy as np
from sklearn import metrics

# 读入预处理好的数据
Train_seq_set = np.load(config.train_sequence_path)
Train_label_set = np.load(config.train_label_path)
Validation_seq = np.load(config.validation_sequence_path)
Validation_label = np.load(config.validation_label_path)

p = Preprocessor()

model1 = TextClassifier(len(config.class_group[0]),'lstm')
model2 = TextClassifier(len(config.class_group[1]),'lstm')
model3 = TextClassifier(len(config.class_group[2]),'lstm')
model4 = TextClassifier(len(config.class_group[3]),'lstm')
model5 = TextClassifier(len(config.class_group[4]),'lstm')
model6 = TextClassifier(len(config.class_group[5]),'lstm')

F1 = []
for i in range(30):
    # 训练模型
    train_seq1, train_label1 = p.shuffle(Train_seq_set[0], Train_label_set[0])
    model1.train(train_seq1, [train_label1[l] for l in range(len(config.class_group[0]))], Validation_seq,
                 [Validation_label[config.class_group[0][i]] for i in range(len(config.class_group[0]))])

    train_seq2, train_label2 = p.shuffle(Train_seq_set[1], Train_label_set[1])
    model2.train(train_seq2, [train_label2[l] for l in range(len(config.class_group[1]))], Validation_seq,
                 [Validation_label[config.class_group[1][i]] for i in range(len(config.class_group[1]))])

    train_seq3, train_label3 = p.shuffle(Train_seq_set[2], Train_label_set[2])
    model1.train(train_seq3, [train_label3[l] for l in range(len(config.class_group[2]))], Validation_seq,
                 [Validation_label[config.class_group[2][i]] for i in range(len(config.class_group[2]))])

    train_seq4, train_label4 = p.shuffle(Train_seq_set[3], Train_label_set[3])
    model1.train(train_seq4, [train_label4[l] for l in range(len(config.class_group[3]))], Validation_seq,
                 [Validation_label[config.class_group[3][i]] for i in range(len(config.class_group[3]))])

    train_seq5, train_label5 = p.shuffle(Train_seq_set[4], Train_label_set[4])
    model1.train(train_seq5, [train_label5[l] for l in range(len(config.class_group[4]))], Validation_seq,
                 [Validation_label[config.class_group[4][i]] for i in range(len(config.class_group[4]))])

    train_seq6, train_label6 = p.shuffle(Train_seq_set[5], Train_label_set[5])
    model1.train(train_seq6, [train_label6[l] for l in range(len(config.class_group[5]))], Validation_seq,
                 [Validation_label[config.class_group[5][i]] for i in range(len(config.class_group[5]))])

    # 保存模型
    model1.save('6lstm_1.npy')
    model2.save('6lstm_2.npy')
    model3.save('6lstm_3.npy')
    model4.save('6lstm_4.npy')
    model5.save('6lstm_5.npy')
    model6.save('6lstm_6.npy')

    # 评估模型
    valid_pred1 = model1.predict(Validation_seq)
    valid_pred2 = model2.predict(Validation_seq)
    valid_pred3 = model3.predict(Validation_seq)
    valid_pred4 = model4.predict(Validation_seq)
    valid_pred5 = model5.predict(Validation_seq)
    valid_pred6 = model6.predict(Validation_seq)
    valid_pred = np.concatenate((valid_pred1, valid_pred2, valid_pred3, valid_pred4, valid_pred5, valid_pred6), axis=0)

    y_pred = [np.argmax(valid_pred[i], axis=1) for i in range(20)]
    y_true = [np.argmax(Validation_label[i], axis=1) for i in range(20)]
    f1 = [metrics.f1_score(y_true[i], y_pred[i], average='micro') for i in range(len(valid_pred))]
    F1.append(f1)
    np.save(config.f1_path+'6lstm.npy')