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

    tmp = p.preprocess_content(Train_raw_data['content'])
    Train_sequences = p.preprocess_text(tmp, train_flag=True)

    tmp = p.preprocess_content(Validation_raw_data['content'])
    Validation_sequences = p.preprocess_text(tmp, train_flag=False)

    tmp = p.preprocess_content(Test_raw_data['content'])
    Test_sequences = p.preprocess_text(tmp, train_flag=False)

    # # +Process Training Input, Validation Input, and Test Input
    # #   1. Divide Content and word Tags
    # #   2. Remove Stop Words
    # #   3. Encoding word Tags
    # #   4. Construct words Dictionary and Padding sequences
    #
    # # -Process Training Input
    # print("Preprocessing Training Data ...")
    # train_content, train_content_tags = p.divide_content_tag(Train_raw_data['content'])
    # train_content, train_content_tags = p.remove_stop_words(train_content, train_content_tags)
    # train_content_tags = p.encode_tag(train_content_tags)
    # train_content, train_content_tags = p.preprocess_text(train_content, train_content_tags)
    #
    # # -Process Validation Input
    # print("Preprocessing Validation Data ...")
    # validation_content, validation_content_tags = p.divide_content_tag(Validation_raw_data['content'])
    # validation_content, validation_content_tags = p.remove_stop_words(validation_content,
    #                                                                   validation_content_tags)
    # validation_content_tags = p.encode_tag(validation_content_tags)
    # validation_content, validation_content_tags = p.preprocess_text(validation_content,
    #                                                                 validation_content_tags,
    #                                                                 train_flag=False)
    # # -Process Test Input
    # print("Preprocessing Test Data ...")
    # test_content, test_content_tags = p.divide_content_tag(Test_raw_data['content'])
    # test_content, test_content_tags = p.remove_stop_words(test_content, test_content_tags)
    # test_content_tags = p.encode_tag(test_content_tags)
    # test_content, test_content_tags = p.preprocess_text(test_content, test_content_tags,
    #                                                     train_flag=False)

    print("Preprocessing Labels ...")
    # #读取所有列标签的名称
    label_names = Train_raw_data.keys().tolist()
    label_names.remove('id')
    label_names.remove('content')
    # label_names.remove('Unnamed: 0')

    #处理标签
    Train_label = []
    Validation_label = []
    Test_label = []

    for name in label_names:
        Train_label.append(p.preprocess_labels(Train_raw_data[name]))
        Validation_label.append(p.preprocess_labels(Validation_raw_data[name]))

    # 数据剪裁
    # Train_sequence, Train_label = p.removeData(Train_sequence, Train_label)

    # 数据复制
    # Train_sequence_set, Train_label_set, train_content_tags = p.replicate_data(train_content, Train_label, train_content_tags)

    # 打乱数据
    # Train_sequence, Train_label = p.shuffle(Train_sequence, Train_label)

    print("Saving Data ...")
    # 保存数据
    np.save(config.train_sequence_path, Train_sequences)
    # np.save(config.train_tags_path, train_content_tags)
    np.save(config.validation_sequence_path, Validation_sequences)
    # np.save(config.validation_tags_path, validation_content_tags)
    np.save(config.test_sequence_path, Test_sequences)
    # np.save(config.test_tags_path, test_content_tags)
    np.save(config.validation_label_path, Validation_label)
    np.save(config.train_label_path, Train_label)

    print("Preprocessed Finished!")