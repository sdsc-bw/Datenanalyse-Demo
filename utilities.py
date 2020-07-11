import pandas as pd 
import numpy as np
import seaborn as sns
# -------------- 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.stats import norm
mpl.rcParams['xtick.labelsize'] = "larger"
mpl.rcParams['ytick.labelsize'] = "larger"
mpl.rcParams['axes.labelsize'] = "larger"
#mpl.rcParams['figure.figsize'] = (10,6) #[6.4, 4.8]
from sklearn.metrics import mean_squared_error

# -------------- 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import preprocessing
import math

import re
import warnings
warnings.filterwarnings("ignore")


def Verteilung_der_Merkmale(df_raw):
    df_raw = df_raw.copy()
    df_raw = df_raw.drop('RIVERSIDE', axis=1)
    feature_cols = set(df_raw.columns) - set(['PREIS'])
    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(16, 12))
    for i, f in enumerate(feature_cols):
        sns.distplot(df_raw[f], ax=axs[int(i/4), i%4]);


def Korrelation(df_raw):
    df_raw = df_raw.copy()
    df_raw = df_raw.drop('RIVERSIDE', axis=1)
    corrmat = df_raw.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
	
def Merkmale_Ziel_Korrelation(df_raw):
    df_raw = df_raw.copy()
    df_raw = df_raw.drop('RIVERSIDE', axis=1)
    feature_cols = set(df_raw.columns) - set(['PREIS'])
    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(20, 12))
    for i, feature in enumerate(feature_cols):
        sns.regplot(x=feature, y='PREIS', data=df_raw, ax=axs[int(i/4), i%4]);
    fig.tight_layout(pad=2.)
		
from sklearn.linear_model import LinearRegression
#import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
lr_regr = LinearRegression()
#xg_regr = xgb.XGBRegressor()
gb_regr = GradientBoostingRegressor()
ls_regr = Lasso()
dt_regr = DecisionTreeRegressor()
rf_regr = RandomForestRegressor()
models = {'LR': lr_regr,'GB':gb_regr,'Lasso':ls_regr,'DT':dt_regr, 'RF': rf_regr}
namesdict =  {'GB' : 'gradient boosting', 'LR' : 'linearen regressions', 
        'Lasso' : 'lasso',
        'DT': 'decision tree',
        'RF' : 'random forest'}

class Regressor:
    def __init__(self, name = "LR"):
        """
        GB : Gradient Boosting
        LR : LinearRegression
        Lasso : Lasso
        DT: Decision Tree
        RF : Random Forest
        """
        self.name = namesdict[name]
        self.model = models[name]
    
    def fit_evaluate(self, X_train, X_test, y_train, y_test, rmse=False):
        
        np.random.seed(123)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        y_pred = pd.Series(pred, name='Vorhersages')
        mse_score = mean_squared_error(y_test, y_pred)
        #print('MSE score: {:d} €', mse_score)
        if rmse:
            rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
            print('Mittlerer quadrierter (test) Fehler: {:,d} €²'.format(int(mse_score)))
            print('Wurzel des mittleren quadrierten (test) Fehlers: {:,d} €'.format(int(rmse_score)))
        else:
            print('(Test) MSE Score: {:,d} €'.format(int(mse_score)))
            
        sns.regplot(y_test, y_pred).set_title('Vorhersage des {} Modells'.format(self.name));
        
        return self #self.model
		
def feature_importance(model, columns):
    plt.figure(figsize=(10,6));
    try:
        feature_importances = pd.DataFrame(model.model.feature_importances_,
                                           index = columns,
                                            columns=['importance']).sort_values('importance', ascending=True);
    except :
        feature_importances = pd.DataFrame(model.model.coef_,
                                           index = columns,
                                            columns=['importance']).sort_values('importance', ascending=True);
    ax = feature_importances.plot.barh(legend=False, title='Feature importance des {} Modells'.format(model.name));  #rot=90)
        