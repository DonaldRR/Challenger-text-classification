# -*- coding:utf-8 -*-

from model import TextClassifier
import config
import Preprocessing
import pandas as pd
import numpy as np

if __name__ == '__main__':
	p = Preprocessing.Preprocessor()
	
	#读取训练集，验证集和测试集s
	Train_raw_data = pd.read_csv(config.train_data_path)
	Validation_raw_data = pd.read_csv(config.validate_data_path)
	Test_raw_data = pd.read_csv(config.test_data_path)

	# #读取所有列标签的名称
	label_names = Train_raw_data.keys().tolist()
	label_names.remove('id')
	label_names.remove('content')

	#预处理训练文本
	tmp = p.preprocess_content(Train_raw_data['content'])
	Train_sequence = p.preprocess_text(tmp, train_flag= True)

	#预处理验证集文本
	tmp = p.preprocess_content(Validation_raw_data['content'])
	Validation_sequence = p.preprocess_text(tmp, train_flag= False)

	#预处理测试集文本
	tmp = p.preprocess_content(Test_raw_data['content'])
	Test_sequence = p.preprocess_text(tmp, train_flag = False)
    
	#处理标签
	Train_label = []
	Validation_label = []
	Test_label = []

	for name in label_names:
		Train_label.append(p.preprocess_labels(Train_raw_data[name]))
		Validation_label.append(p.preprocess_labels(Validation_raw_data[name]))

	np.save(config.train_sequence_path, Train_sequence)
	np.save(config.validation_sequence_path, Validation_sequence)
	np.save(config.test_sequence_path, Test_sequence)
	np.save(config.train_label_path, Train_label)
	np.save(config.validation_label_path, Validation_label)

F1 = []
for i in range(30):
	model = TextClassifier(nn_type='cnn')
	model.train(Train_sequence, Train_label, Validation_sequence, Validation_label)

	valid_pred = model.predict(Validation_sequence)
	y_pred = []
	y_true = []
	for i in range(len(label_names)):
		y_pred.append(np.argmax(y_pred[i], axis=1))
		y_true.append(np.argmax(Validation_label[i], axis=1))
	f1 = model.evaluate(y_pred, y_true)
	F1.append(f1)
	print('### F1 score:{} ###'.format(f1))
	print('### Avg F1:{} ###'.format(np.average(f1)))
	np.save(F1)

	model.save('cnn')

	# clf = model.TextClassifier()