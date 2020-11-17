import numpy as np 
import pandas as pd
import csv
from random import random

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.svm import SVR
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


dataset = "C:/Users/Gianluca/Desktop/SmartDobot/"

pixel_distanza = pd.read_csv(dataset + 'pixel_per_distanza.csv')
dobot_distanza = pd.read_csv(dataset + 'dobot_per_distanza.csv')

rotazioni_oggetti = pd.read_csv(dataset + 'oggetti_rotazione.csv')
rotazioni_Dobot = pd.read_csv(dataset + 'rotazione_Dobot.csv')

def crea_regressore_x():
    train_for_x = pixel_distanza.iloc[0:]
    label_for_x = dobot_distanza.iloc[0:,0]
    #print(label_for_x)
    #print(train_for_x)
    
    degree=3
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train_for_x, label_for_x)
    pred_poly = polyreg.predict(train_for_x)
    rmse = np.sqrt(mean_squared_error(label_for_x,pred_poly))
    #print('RMSE poly reg: ', rmse)
    
    model_Elastic = ElasticNetCV()
    model_Elastic.fit(train_for_x, label_for_x)
    pred_Elastic = model_Elastic.predict(train_for_x)
    rmse = np.sqrt(mean_squared_error(label_for_x,pred_Elastic))
    #print('RMSE elastic: ', rmse)
  
    model_Tree = DecisionTreeRegressor()
    model_Tree.fit(train_for_x, label_for_x)
    pred_Tree = model_Tree.predict(train_for_x)
    rmse = np.sqrt(mean_squared_error(label_for_x,pred_Tree))
    #print('RMSE tree: ', rmse)
    
    svr = SVR(kernel='rbf', C=100, gamma='auto')
    svr.fit(train_for_x, label_for_x)
    pred_svr = svr.predict(train_for_x)
    rmse = np.sqrt(mean_squared_error(label_for_x,pred_svr))
    #print('RMSE svr: ', rmse)

    return model_Elastic

def x_predizione(model, val1,val2):
    lista = []
    lista = lista + [val1, val2]
    file='Temp/test'+str(random())+'.csv'
    with open(file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, escapechar='', quotechar=' ')
        wr.writerow(lista)
    test = pd.read_csv(file, header=None)
    predizione = model.predict(test)
    #print('X', predizione)
    return predizione

def crea_regressore_y():
    train_for_x = pixel_distanza.iloc[0:]
    label_for_x = dobot_distanza.iloc[0:,1]
    #print(label_for_x)
    #print(train_for_x)

    degree=3
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train_for_x, label_for_x)
    
    model_Y = ElasticNetCV()
    model_Y.fit(train_for_x, label_for_x)
    
    #r_sq = model_Y.score(train_for_x, label_for_x)
    #print('coefficient of determination Y:', r_sq)
    return model_Y

def y_predizione(model, val1, val2):
    lista = []
    lista = lista + [val1, val2]
    file='Temp/test'+str(random())+'.csv'
    with open(file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, escapechar='', quotechar=' ')
        wr.writerow(lista)
    test = pd.read_csv(file, header=None)
    predizione = model.predict(test)
    #print('Y', predizione)

    return predizione

def crea_regressore_r():
    train_for_x = rotazioni_oggetti.iloc[0:]
    label_for_x = rotazioni_Dobot.iloc[0:]
    #print(label_for_x)
    #print(train_for_x)

    degree=3
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train_for_x, label_for_x)

    model_R = ElasticNetCV()
    model_R.fit(train_for_x, label_for_x)
    
    #r_sq = model_R.score(train_for_x, label_for_x)
    #print('coefficient of determination R:', r_sq)
    return model_R

def r_predizione(model, val1):
    lista = []
    lista = lista + [val1]
    file='Temp/test'+str(random())+'.csv'
    with open(file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, escapechar='', quotechar=' ')
        wr.writerow(lista)
    test = pd.read_csv(file, header=None)
    predizione = model.predict(test)
    #print('R', predizione)

    return predizione


