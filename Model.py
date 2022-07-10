#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: T.Tunç Kulaksız
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error


veriler = pd.read_csv('preprocessing_car.csv')




len(veriler)*0.01
veriler= veriler.sort_values("4",ascending=False).iloc[90:]
veriler= veriler.sort_values("4",ascending=True).iloc[90:]
veriler= veriler.sort_values("5",ascending=False).iloc[90:]
veriler= veriler.sort_values("5",ascending=True).iloc[90:]
veriler= veriler.sort_values("7",ascending=False).iloc[90:]
veriler= veriler.sort_values("7",ascending=True).iloc[90:]
veriler= veriler.sort_values("8",ascending=False).iloc[90:]
veriler= veriler.sort_values("8",ascending=True).iloc[90:]
veriler= veriler.sort_values("13",ascending=False).iloc[90:]
veriler= veriler.sort_values("13",ascending=True).iloc[90:]
veriler= veriler.sort_values("14",ascending=False).iloc[90:]
veriler= veriler.sort_values("14",ascending=True).iloc[90:]
print(veriler.corr())


veriler1 = veriler.iloc[:,[4,5,7,8,13,14,15]]

X = veriler1.iloc[:,0:6].values
Y = veriler1.iloc[:,6].values

X=np.asarray(X).astype(np.float32)
Y=np.asarray(Y).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,BatchNormalization,LeakyReLU 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

model = Sequential()


model.add(Dense(1024,input_dim = X_train.shape[1]))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(512))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.01))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.05))

model.add(Dense(1, activation="linear"))

optimizer = tensorflow.keras.optimizers.Adam(lr=0.005, decay=5e-4)

model.compile(optimizer =optimizer, loss='mean_absolute_error')

checkpoint_name = 'Weights\Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

model.fit(X_train,y_train, validation_data=(X_test,y_test),callbacks=callbacks_list, batch_size = 1024, epochs = 500, verbose=1 )

y_pred = model.predict(X_test)

print("yanılma payı")
print(mean_absolute_error(y_test,y_pred))

print("r^2 score")
print(r2_score(y_test,y_pred))

"""
model.save("pred_car.h5")
"""




