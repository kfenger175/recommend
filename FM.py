from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np

train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]
#构造向量化类 https://www.cnblogs.com/hufulinblog/p/10591339.html
v = DictVectorizer()
#对非数值类型进行one-hot编码，数值类型不变
X = v.fit_transform(train)
print(X.toarray())

y = np.repeat(1.0, X.shape[0])
print(y)

fm = pylibfm.FM()
fm.fit(X,y)
test = fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print(test)
print("success!")
#-------------以上为简单测试代码#

def loadData(filname,path = "ml-100k/"):
    data = []
    y = []
    users = set()
    items = set()
    with open(path+filname) as f:
        for line in f:
            (user,movieid,rating,ts) = line.split('\t')
            data.append({"user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)
    return (data,np.array(y),users,items)

#导入训练集和测试集
(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")

v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)
#训练模型并测试
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)

#预测结果并打印误差
preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print("FM MSE: %.4f" % mean_squared_error(y_test,preds))