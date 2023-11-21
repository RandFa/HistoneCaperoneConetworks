## importing libraries
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import numpy as np
from statsmodels.formula.api import glm
from sklearn.decomposition import PCA
import plotly.express as px
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, mean_absolute_error
import random
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import scanpy as sc
from patsy.builtins import *
from scipy import stats
import statsmodels
from scipy.stats.mstats import spearmanr
import conorm
import inspect
from matplotlib.transforms import Affine2D


## setting the working directory 
os.chdir('./Dropbox (Institut Curie)/Sebastien Lemaire/Rand_Fatouh')

## reading the expression dataframe 
df = pd.read_csv("./Data/GTExTCGAsmall.csv",index_col=0)

## reading meta data 
meta = pd.read_csv('./Data/GTExTCGAsmallmeta.csv', index_col=0)

## reading histones/chaperones gene list
hist = pd.read_csv('./Data/histone_chaperone.csv',index_col=0)

## filtering expression matrix for h/c gene only 
dfhist = pd.merge(hist[['GeneName']], df,left_index=True, right_index=True)
dfhist.drop('GeneName', inplace=True, axis=1)

## setting up the dataset for analysis by transposing
dft = dfhist.T
dft2 = df.T

##splitting gtex and tcga - h/c and the whole dataset (all)
dfGTEx = dft.iloc[:5605,:]
dfGTExall = dft2.iloc[:5605,:]
dfTCGA = dft.iloc[5605:,:]
dfTCGAall = dft2.iloc[5605:,:]
ygtex = meta.iloc[:5605,:][["spl_organ"]]
ytcga = meta.iloc[5605:,:][["spl_organ"]]


## setting the random seed befor rf to get reproducable results
random.seed(10)

## getting the list of tissues
tissues = list(ygtex['spl_organ'].unique())


## the random forest model and saving the figures
for t in tissues:
    ygtex[t] = np.where(ygtex['spl_organ'] == t, t, 'other')
    X_train, X_test, y_train, y_test = train_test_split(dfGTExfiltered, ygtex[t], test_size=0.2, random_state=0, stratify=ygtex[t])
    rf = BalancedRandomForestClassifier(random_state=0, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf.feature_importances_
    sorted_idx = rf.feature_importances_.argsort()[::-1]
    plt.clf()
    plt.barh(X_train.columns[sorted_idx][:20], rf.feature_importances_[sorted_idx][:20])
    plt.xlabel("Random Forest Feature Importance")
    plt.savefig('./Results/GTEx/RF/' + t + 'importancefiltered.png', bbox_inches='tight')
    df1 = hist[hist.GeneOfficialSymbol.isin(X_train.columns[sorted_idx])]
    df2 = pd.DataFrame({'Genes' : X_train.columns[sorted_idx], 'Importance' : rf.feature_importances_[sorted_idx]})
    df2.set_index('Genes', inplace= True)
    dfimp = pd.merge(df1, df2, left_index=True, right_index=True)
    dfimp.to_csv('./Results/GTEx/RF/' + t + 'importancefiltered.csv', encoding='utf-8', index=True)
    plt.clf()
    confusion_matrix = metrics.confusion_matrix(y_test, rf.predict(X_test))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [t, 'Other'])
    cm_display.plot()
    plt.savefig('./Results/GTEx/RF/' + t + 'confusion_matrix.png', bbox_inches='tight')
