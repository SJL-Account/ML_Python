
import pandas as pd
import numpy as np
from  sklearn.preprocessing import  OneHotEncoder
from  sklearn.preprocessing import  StandardScaler
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

class my_data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.i = 0
        self.load_data()

    def load_data(self):

        x_train, x_test, y_train, y_test=self.train_test_split()


        self.x_data = x_train
        self.y_data = y_train

    def train_test_split(self):

        data = pd.read_csv('all_data.txt', delimiter='\t',encoding='gbk')

        #data=shuffle(data)

        test_data = data[data['训练测试分区'] == 0]
        train_data = data[data['训练测试分区'] != 0]
        x_train = train_data[['AC', 'CNL', 'DEN', 'GR', 'PE', 'RLLD']]
        y_train = train_data['code']

        x_test = test_data[['AC', 'CNL', 'DEN', 'GR', 'PE', 'RLLD']]
        y_test = test_data['code']

        ohe = OneHotEncoder()
        std = StandardScaler()
        x_train=std.fit_transform(X=x_train)
        y_train=ohe.fit_transform(np.matrix(y_train).T).toarray()
        x_test=std.fit_transform(X=x_test)
        y_test=ohe.fit_transform(np.matrix(y_test).T).toarray()

        return x_train, x_test, y_train, y_test

    def next_batch(self):
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return re_x, re_y

    def __len__(self):
        return int(len(self.x_data) / self.batch_size)

    def __getitem__(self, i):
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return (re_x, re_y)

    def __next__(self):
        if self.i == len(self):
            raise StopIteration
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return re_x, re_y

    def __iter__(self):
        return self

