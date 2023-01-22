# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:58:02 2023

@author: Murat
"""

#1.Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.Data Preprocessing
#2.1 Data Boot
veriler= pd.read_csv("veriler.csv")
#pd.read_csv()
print(veriler)

boy=veriler[["boy"]]
print(boy)

boykilo=veriler[["boy","kilo"]]
print(boykilo)


#Missing Data

#sci-kit learn
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

Yas=veriler.iloc[:,1:4].values #integer values
print(Yas)

imputer=imputer.fit(Yas[:,1:4]) #fit=öğretmek
Yas[:,1:4]=imputer.transform(Yas[:,1:4]) #transform= öğrenmek
print(Yas)

#Tnsition from Categorical Data to Numerical Data (Data Type Conversion)
#encoder: Kategorik--> Numeric
ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

# DataFrame Conversion of Numpy Arrays

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)

#Dataframe Merge Process
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#Division of the Dataset for Training and Testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#Scaling of Data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
















