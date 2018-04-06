#https://www.kaggle.com/mlg-ulb/creditcardfraud

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df_full = pd.read_csv('creditcard.csv')
print(df_full.head())


df_full.sort_values(by='Class', ascending=False, inplace=True)
df_full.drop('Time', axis=1,  inplace = True)

print(df_full.head())



df_sample = df_full.iloc[:3000,:]
df_sample.Class.value_counts()

feature = np.array(df_sample.values[:,0:29])
label = np.array(df_sample.values[:,-1])


from sklearn.utils import shuffle

shuffle_df = shuffle(df_sample)

df_train = shuffle_df[0:2400]
df_test = shuffle_df[2400:]

train_feature = np.array(df_train.values[:,0:29])
train_label = np.array(df_train.values[:,-1])
test_feature = np.array(df_test.values[:,0:29])
test_label = np.array(df_test.values[:,-1])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train_feature)
train_feature_trans = scaler.transform(train_feature)
test_feature_trans = scaler.transform(test_feature)

from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt 
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

model = Sequential()

model.add(Dense(units=200, 
                input_dim=29, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=200,  
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))

print(model.summary()) 

model.compile(loss='binary_crossentropy',  
              optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=train_feature_trans, y=train_label,  
                          validation_split=0.8, epochs=200, 
                          batch_size=500) 


show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')



scores = model.evaluate(test_feature_trans, test_label)
print('\n')
print('accuracy=',scores[1])


prediction = model.predict_classes(test_feature_trans)

df_ans = pd.DataFrame({'Real Class' :test_label})
df_ans['Prediction'] = prediction

df_ans[ df_ans['Real Class'] != df_ans['Prediction'] ]
