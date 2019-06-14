import jieba
import codecs
import re
import numpy as np
import heapq
import pickle
import tensorflow.contrib.keras as kr

re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # 正则项

filename = "I:\python project\multimodal_emotion/text_rnn\datasets\cnews.test.txt"

text = []
label = []

with codecs.open(filename, 'r', encoding='utf-8') as f:
    for _, line in enumerate(f):
        line = line.strip()
        line = line.split('\t')
        assert len(line) == 2
        blocks = re_han.split(line[1])
        word = []
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        text.append(word)
        label.append(line[0])
    f.close()

# 将text的文本形式转换为词向量

with open("I:\python project\multimodal_emotion/text_rnn\emb_mat_dict.pkl", 'rb') as f:
    embed_mat = pickle.load(f)

text_in_vec = []

for t in text:
    temp = []
    for w in t:
        temp.append(embed_mat[w])
    temp = np.reshape(np.asarray(temp, dtype=np.float32), [1, -1, 100])
    text_in_vec.append(temp)
    pass

categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
cat_to_id = dict(zip(categories, range(len(categories))))    # 在python中用字典做映射比if语句快的多

label_id = []
for l in label:
    label_id.append(cat_to_id[l])
    pass

y_onehot = kr.utils.to_categorical(label_id, num_classes=10)

# y_onehot是文本分类标签，text_in_vec是以词向量为形式的文本数据。


def data_input():
    return text_in_vec, y_onehot
