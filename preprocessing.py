#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: T.Tunç Kulaksız
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler0 = pd.read_csv('car.csv')
veriler = veriler0.values


from sklearn import preprocessing

le= preprocessing.LabelEncoder()
veriler[:,0] = le.fit_transform(veriler[:,0])

le= preprocessing.LabelEncoder()
veriler[:,1] = le.fit_transform(veriler[:,1])

le= preprocessing.LabelEncoder()
veriler[:,2] = le.fit_transform(veriler[:,2])

le= preprocessing.LabelEncoder()
veriler[:,3] = le.fit_transform(veriler[:,3])

le= preprocessing.LabelEncoder()
veriler[:,4] = le.fit_transform(veriler[:,4])

le= preprocessing.LabelEncoder()
veriler[:,5] = le.fit_transform(veriler[:,5])

le= preprocessing.LabelEncoder()
veriler[:,6] = le.fit_transform(veriler[:,6])

le= preprocessing.LabelEncoder()
veriler[:,7] = le.fit_transform(veriler[:,7])

le= preprocessing.LabelEncoder()
veriler[:,8] = le.fit_transform(veriler[:,8])

le= preprocessing.LabelEncoder()
veriler[:,9] = le.fit_transform(veriler[:,9])

le= preprocessing.LabelEncoder()
veriler[:,10] = le.fit_transform(veriler[:,10])

le= preprocessing.LabelEncoder()
veriler[:,11] = le.fit_transform(veriler[:,11])

le= preprocessing.LabelEncoder()
veriler[:,12] = le.fit_transform(veriler[:,12])


veriler = pd.DataFrame(veriler)
veriler.to_csv('preprocessing_car.csv')








