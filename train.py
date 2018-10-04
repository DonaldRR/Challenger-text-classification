from model import TextClassifier
import numpy as np
import config

# 读入预处理好的数据
Train_seq = np.load(config.train_sequence_path)
Train_label = np.load(config.train_label_path)
Validation_seq = np.load(config.validation_sequence_path)
Validation_label = np.load(config.validation_label_path)

F1 = []
model = TextClassifier('lstm')
for i in range(30):
    # 训练模型
    model.train(Train_seq, Train_label, Validation_seq, Validation_label)

    # 评估模型
    f1 = model.evaluate(model, Validation_seq, Validation_label)
    F1.append(f1)
    np.save(config.f1_path+'lstm.npy')

    # 保存模型
    model.save('lstm')
