import pandas as pd
import numpy as np

# Read dataset into X and Y
df = pd.read_csv('~/HousePrices_Regression/train-v01.csv', delim_whitespace=True, header=None)
dataset = df.values
X_train = dataset[:, 0:21]
y_train = dataset[:, 21]


df2 = pd.read_csv('~/HousePrices_Regression/test-v01.csv', delim_whitespace=True, header=None)
dataset2 = df2.values
X_test = dataset2[:, 0:21]

df3 = pd.read_csv('~/HousePrices_Regression/valid-v01.csv', delim_whitespace=True, header=None)
dataset3 = df3.values
X_valid = dataset3[:, 0:21]
y_valid = dataset3[:, 21]


def normalize(train, valid, test):
#    tmp = np.concatenate((train,valid),axis=0)
    tmp = train
    mean, std = tmp.mean(axis=0), tmp.std(axis=0)
    print("tmp.shape= ", tmp.shape)
    print("mean.shape= ", mean.shape)
    print("std.shape= ", std.shape)
    print("mean= ", mean)
    print("std= ", std)
    train = (train - mean) / std
    valid = (valid - mean) / std
    test = (test - mean) / std
    return train, valid, test
    
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.layers.core import Dense, Activation

X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)

model = Sequential()
model.add(Dense(40, input_dim=21, init='normal', activation='relu'))
model.add(Dense(80, input_dim=40, init='normal', activation='relu'))
model.add(Dense(60, input_dim=80, init='normal', activation='relu'))
model.add(Dense(30, input_dim=60, init='normal', activation='relu'))
model.add(Dense(18, input_dim=30, init='normal', activation='relu'))
# No activation needed in output layer (because regression)
model.add(Dense(1, init='normal'))

# Compile Model
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mae', optimizer='adam')



model.fit(X_train, y_train, epochs=200, batch_size=32,verbose=1)


score=model.evaluate(X_valid, y_valid, batch_size=32)


pred=model.predict(X_test)