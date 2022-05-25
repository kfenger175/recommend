from turtle import shape
import pandas as pd
import numpy as np

from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#显示python进度条
from tqdm import tqdm

# dense特征取对数　　sparse特征进行类别编码
# (类别编码一般用于标签编码，特征很少用到)
def process_feat(data, dense_feats, sparse_feats):
    df = data.copy()
    #dense
    df_dense = df[dense_feats].fillna(0.0) #使用0来填充缺失值
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1 + x) if x > -1 else -1) #
    
    #sparse
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])
    
    df_sparse_arr = []
    for f in tqdm(sparse_feats):
        #get_dummies pandas中one hot编码
        data_new = pd.get_dummies(df_sparse.loc[:,f])
        #dataframe.columns返回列标签 dataframe.index返回行标签
        data_new.columns = [f + "_{}".format(i) for i in range(data_new.shape[1])]
        df_sparse_arr.append(data_new)

    df_new = pd.concat([df_dense] + df_sparse_arr,axis = 1)
    return df_new

#FM 特征组合层

class crossLayer(layers.Layer):
    def __init__(self, input_dim, output_dim = 10, **kwargs) -> None:
        super(crossLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        #定义交叉特征的权重
        self.kernel = self.add_weight(name = "kernel",
                                      shape = (self.input_dim,self.output_dim),
                                      initializer = "glorot_uniform",
                                      trainable = True)
    def call(self, x): 
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return 0.5 * K.mean(a - b, 1, keepdims = True)

def FM(feature_dim):
    inputs = Input(shape = (feature_dim,))

    #一阶特征
    linear = Dense(units=1,
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(inputs)

    #二阶特征
    cross = crossLayer(feature_dim)(inputs)
    add = Add()([linear, cross]) #将一阶特征和二阶特征相加构建FM模型

    pred = Dense(units = 1, activation = "sigmoid")(add)
    model = Model(inputs = inputs, outputs = pred)

    model.summary()
    model.compile(loss = 'binary_crossentropy',
                 optimizer = optimizers.Adam(),
                 metrics=['binary_accuracy'])
    return model



# 读取数据
print('loading data...')
data = pd.read_csv('./data/kaggle_train.csv')

# dense 特征开头是I，sparse特征开头是C，Label是标签
cols = data.columns
#print(type(cols))  # <class 'pandas.core.indexes.base.Index'>
#print(type(data.columns.values)) #<class 'numpy.ndarray'>
#获取dense和sparse特征对应的列标签list
dense_feats = [f for f in cols if f[0] == 'I']
sparse_feats = [f for f in cols if f[0] == 'C']

print(dense_feats)
print(sparse_feats)

# 对dense数据和sparse数据分别处理
print('processing features')
feats = process_feat(data, dense_feats, sparse_feats)

# 划分训练和验证数据
x_trn, x_tst, y_trn, y_tst = train_test_split(feats, data['Label'], test_size=0.2, random_state=2020)

# 定义模型
model = FM(feats.shape[1])

# 训练模型
model.fit(x_trn, y_trn, epochs=10, batch_size=128, validation_data=(x_tst, y_tst))