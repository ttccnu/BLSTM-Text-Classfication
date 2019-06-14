import numpy as np
import pickle as pk
import tensorflow as tf
from text_rnn.model import RNN
import random
from text_rnn.text_preprocess import data_input

data, label = data_input()

rand_id = np.arange(len(label))
random.shuffle(rand_id)

data = [data[i] for i in rand_id]
label = [label[i] for i in rand_id]

# ----------------------------------------------------------
train_data = data[1000:]   # 训练数据
train_label = label[1000:]
test_data = data[0:1000]  # 测试数据
test_label = label[0:1000]
# -----------------------------------------------------------

print("开始训练")
rnn = RNN()
rnn.blstm(train_data, train_label, test_data, test_label)

