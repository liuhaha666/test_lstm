#coding=utf-8

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D,MaxPooling1D,GlobalMaxPool1D
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import Masking
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import keras
import jieba as jb
import re
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from keras.models import load_model
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
#from wiki_one import eda_func
import csv
import random
from keras import optimizers
# print('----------数据增广---开始---------')
# df = pd.read_csv('C:/Users/Lenovo/Desktop/online_shopping_10_cats/online_shopping_10_cats.csv')
# print ('type df is ',type(df))
#
# df=df[['cat','review']]
# print ('type df is 1 ',type(df))
# print("数据总量: %d ." % len(df))
# print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
# print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
# df[df.isnull().values==True]
# df = df[pd.notnull(df['review'])]
# d = {'cat':df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
# df_cat = pd.DataFrame(data=d).reset_index(drop=True)
# print ('df_cat before zengguang is',df_cat)
# def write_detail(lst_in,lable):
#     lis_detail = []
#     for str_i in lst_in:
#         #print (str_i)
#         lis_detail.append(lable)
#         lis_detail.append(0)
#         lis_detail.append(str_i)
#     #print ('lis_detail is',lis_detail)
#     n = 3  # 大列表中几个数据组成一个小列表
#     rows = ([lis_detail[i:i + n] for i in range(0, len(lis_detail), n)])
#     #print ('rows is',rows)
#     header = ['cat', 'label', 'review']
#     out = open('C:/Users/Lenovo/Desktop/online_shopping_10_cats//online_shopping_10_cats.csv', 'a', newline='',encoding='utf-8')
#     csv_writer = csv.writer(out, dialect='excel')
#     #csv_writer.writerow(header)
#     csv_writer.writerows(rows)
#     out.close()
# num = 0
# for cat_lable in df_cat['cat']:
#     d_content = df[df['cat']==cat_lable]
#     d_review = d_content['review'].values
#     zeng_guang_pd = df_cat['count'][num]
#     zeng_guang_count = 5000-zeng_guang_pd
#     print ('----')
#     if zeng_guang_pd > 1500:
#         print ('>1500')
#         #随机选500条，曾广4倍
#         for j in range(500):
#             rand_int = random.randint(0,1500)
#             #print('rand_int is',rand_int)
#             #print (eda_func(sentence=d_review[rand_int]))
#             list_ret = eda_func(sentence=d_review[rand_int],num_aug=4)[:]
#             print ('list_ret is',list_ret)
#             write_detail(list_ret,cat_lable)
#     elif 700 < zeng_guang_pd < 1500:
#         print ('700< n <1500')
#         #全部曾广4倍
#         for i in range(500):
#             rand_int = random.randint(0, 700)
#             list_ret = eda_func(sentence=d_review[rand_int],num_aug=4)[:]
#             print('list_ret is', list_ret)
#             write_detail(list_ret, cat_lable)
#     else:
#         print ('<700')
#         #增广八倍
#         for i in range(len(d_review)):
#             list_ret = eda_func(sentence=d_review[i],num_aug=8)[:]
#             print('list_ret is', list_ret)
#             write_detail(list_ret, cat_lable)
#     num += 1
# print ('------------------after')
df = pd.read_csv('C:/Users/Lenovo/Desktop/online_shopping_10_cats/online_shopping_10_cats.csv')
df=df[['cat','review']]
print("数据总量: %d ." % len(df))
df[df.isnull().values==True]
df = df[pd.notnull(df['review'])]
d = {'cat':df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
#获得类别标签,本项目中为0-9
df['cat_id'] = df['cat'].factorize()[0]
#df[ cat_id] shape is shape (15116,)
#标签去重，统计总类别数量,目前项目中有10类
cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
print ('cat_id_df is',cat_id_df)
#cat_to_id = dict(cat_id_df.values)
#类别转化为字典
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)
# # 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
	line = str(line)
	if line.strip() == '':
		return ''
	rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
	line = rule.sub('', line)
	return line

def stopwordslist(filepath):
	stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
	return stopwords
# #
# 加载停用词
stopwords = stopwordslist("C:/Users/Lenovo/Desktop/online_shopping_10_cats/stopword.txt")
#删除除字母,数字，汉字以外的所有符号
df['clean_review'] = df['review'].apply(remove_punctuation)
#分词，并过滤停用词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
#print ('df.head is',df.head())
# 设置最频繁使用的5000个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)
MAX_NB_WORDS = 5000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 64
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 80
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#首先用Tokenizer的 fit_on_texts 方法学习出文本的字典
#df[cut_review]，shape为(13718,)
#df[cut_review] 类型是series，series类似于列表，但是有索引
#df[cut_review].values 类型是numpy.ndarray，shape为(13718,)
#df[cut_review].values中存储的是所有文本分词之后形成的矩阵
#fit_on_text(texts) 使用一系列文档来生成token词典，
# texts中，每个元素为一个文档。
# print ('here is',(df['cut_review'].values))
# print ('h is',(df['cut_review']))
tokenizer.fit_on_texts(df['cut_review'].values)
#下面一条打印语句打印结果为None
#print ('tokenizer.fit_osn_texts(df[cut_review].values) is',tokenizer.fit_on_texts(df['cut_review'].values))
#word_index 就是对应的单词和数字的映射关系dict
word_index = tokenizer.word_index
#print ('word_index is',word_index)
#print('共有 %s 个不相同的词语.' % len(word_index))
#print ('df[cut_review] is',df['cut_review'])
#X为一个list，其中的每个元素是每个样本（每个句子）转换成的词向量
#X长度为所有句子的个数。在本例中为13718
#通过这个dict可以将每个string的每个词转成数字，可以用texts_to_sequences
#texts_to_sequences(texts)
# 将多个文档转换为word下标的向量形式,
# shape为[len(texts)，len(text)]
# -- (文档数，每条文档的长度)
X = tokenizer.texts_to_sequences(df['cut_review'].values)
#print ('X b is',X)
#填充前X内无零值

#填充X,让X的各个列的长度统一
#X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH,padding='pre')
#print ('X a is',X)
#多类标签的预处理
# pd.get_dummies可以将一个特征的类别抽取出来当作新的特征使用，
# 这样增加了样本的维度，是一种常用的数据预处理方法。
Y = pd.get_dummies(df['cat_id']).values
#df['cat_id'].values长度为13718，值为每个样本对应的类的标记
#Y shape (13718, 10)，值为每个样本对应的类别。
#print ('Y is',Y)
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import numpy as np
from keras.utils import plot_model
INPUT_DIM = 64
TIME_STEPS = 64
SINGLE_ATTENTION_VECTOR = False
#from attention_utils import get_activations, get_data_recurrent
#拆分训练集和测试集
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print('inputs.shape is',inputs.shape)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    print('a.shape after permute is',a.shape)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    print('a.shape after dense is', a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
#定义模型
# print ('X_train is',X_train)
# print ('Y_train is',Y_train)
# print ('X_test is',X_test)
# print ('Y_test is',Y_test)
#lstm训练模型部分代码--开始
from keras.callbacks import TensorBoard
from keras.layers import LeakyReLU,ReLU,ELU

#keras.layers.advanced_activations.
tbCallBack = TensorBoard(log_dir='D:/BaiduNetdiskDownload/TextCNN_models/logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
def lstm_with_attention():
    main_input = Input(shape=(TIME_STEPS,))
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(MAX_NB_WORDS, 64)(main_input)
    # attention before lstm
    # attention_mul = attention_3d_block(embedder)
    lstm_out = LSTM(100, return_sequences=True,
                    kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                    bias_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(0.00001),
                    #activity_regularizer=regularizers.l1(0.0001),
                    bias_regularizer=regularizers.l2(0.00001)
                    )(embedder)
    my_leaky = LeakyReLU(alpha=0.001)(lstm_out)
    #attention after lstm
    attention_mul = attention_3d_block(my_leaky)
    attention_mul = Flatten()(attention_mul)
    batch = BatchNormalization()(attention_mul)
    drop = Dropout(0.3)(batch)
    main_output = Dense(10, activation='softmax',
                        kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                        bias_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                        kernel_regularizer=regularizers.l2(0.00001),
                        #activity_regularizer=regularizers.l1(0.01),
                        bias_regularizer=regularizers.l2(0.00001)
                        )(drop)
    model = Model(inputs=main_input, outputs=main_output)
    my_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,clipvalue=0.5)

    model.compile(loss='categorical_crossentropy', optimizer=my_adam, metrics=['accuracy'])
    return model
model = lstm_with_attention()
print(model.summary())
epochs = 2
batch_size = 64

localtime = time.asctime( time.localtime(time.time()) )
print ('--',localtime)
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[tbCallBack])
# # #lstm训练模型部分代码--结束
# # # #对测试集运算
model.save('D:/BaiduNetdiskDownload/TextCNN_models/lstm_att_after.h5')

model_use = load_model('D:/BaiduNetdiskDownload/TextCNN_models/lstm_att_after.h5')
j = plot_model(model_use,to_file='D:/BaiduNetdiskDownload/TextCNN_models/lstm_att_after.png',show_shapes=True,show_layer_names=False)

train_result = model_use.predict(X_train)
train_labels = np.argmax(train_result, axis=1)
print('train_labels shape is',train_labels.shape)
y_train_labels = np.argmax(Y_train, axis=1)
print('y_train_labels shape is',y_train_labels.shape)
print('lstm train 准确率',metrics.accuracy_score(y_train_labels, train_labels))
accr = model_use.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#加载已训练好的lstm模型进行预测
#训练时屏蔽下面两行代码
#训练后预测时，放开下面两行代码
y_pred = model_use.predict(X_test)
# print ('y_pred b is',y_pred)
# # print ('Y_test b is',Y_test)
#使用CNN训练时屏蔽以下两行。
y_pred = y_pred.argmax(axis = 1)
Y_test = Y_test.argmax(axis = 1)
#得到精确率、召回率、F1值
from sklearn.metrics import classification_report
print('15 accuracy %s' % accuracy_score(y_pred, Y_test))
print(classification_report(Y_test, y_pred, target_names=cat_id_df['cat'].values))
################
#model predict
model_use_lstm = load_model('D:/BaiduNetdiskDownload/TextCNN_models/bilstm.h5')
startTimehere = time.time()
def lstm_predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=250)
    pred = model_use_lstm.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
    return cat_id_df[cat_id_df.cat_id==cat_id]['cat'].values[0]
print(lstm_predict('可能中外文化理解差异，可能与小孩子太有代沟，不觉得怎么样，还是比较喜欢水墨画一点风格的漫画书，但愿我女儿长大一点能喜欢（22个月中）'))
print('---predict Completed.Took %f s.' % (time.time() - startTimehere))
