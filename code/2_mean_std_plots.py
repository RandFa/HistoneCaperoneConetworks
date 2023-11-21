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

##calculating mean and standard deviation
avgGTEx = np.mean(dfGTEx, axis=0)
stdGTEX = dfGTEx.std(axis = 0)
avgallGTEx = np.mean(dfGTExall, axis=0)
stdallGTEx = dfGTExall.std(axis = 0)
avgTCGA = np.mean(dfTCGA, axis=0)
stdTCGA = dfTCGA.std(axis = 0)
avgallTCGA = np.mean(dfTCGAall, axis=0)
stdallTCGA = dfTCGAall.std(axis = 0)

## replacing gene codes with names for reading the results (h/c genes)
dfGTEx.columns = hist['GeneOfficialSymbol'].loc[dfGTEx.columns]
dfTCGA.columns = hist['GeneOfficialSymbol'].loc[dfTCGA.columns]

## mean and standard deviation scatter plot of GTEx
plt.clf()
plt.scatter(avgallGTEx, stdallGTEx, s = 1, c='black', label = 'All genes')
plt.scatter(avgGTEx, stdGTEX, s = 3, c='red', label='H/C genes')
plt.axvline(x=7, color='r', label='mean = 7')
plt.axhline(y=1.2, color='r', label='std =1.2')
plt.title("Mean vs STD")
plt.xlabel("Mean")
plt.ylabel("Std")
plt.legend(loc='upper right')
plt.savefig('./Results/GTEx/meanvsstd.png')

## mean density plot - GTEx
plt.clf()
avgallGTEx.plot.density(label = 'All genes',bw_method=0.1, color='black')
plt.axvline(x=7, color='r', label='mean = 7')
plt.xlim(0,25) 
plt.xticks(np.arange(-1, 26, 2.5))
plt.title('Density Plot for the mean')
plt.legend(loc='upper right')
plt.xlim([0, 21])
plt.savefig('./Results/GTEx/meandensity.png', bbox_inches='tight')

## standard deviation density plot - GTEx
plt.clf()
stdallGTEx.plot.density(label = 'All genes',bw_method=0.1, color='black')
plt.axvline(x=1.2, color='r', label='std =1.2')
plt.xlim(0,6) 
plt.xticks(np.arange(0, 7,1))
plt.title('Density Plot for the standard deviation')
plt.legend(loc='upper right')
plt.xlim([0, 5.5])
plt.savefig('./Results/GTEx/stddensity.png', bbox_inches='tight')

## mean and standard deviation scatter plot of TCGA
plt.clf()
plt.scatter(avgallTCGA, stdallTCGA, s = 1, c='black', label = 'All genes')
plt.scatter(avgTCGA, stdTCGA, s = 3, c='red', label='H/C genes')
plt.axvline(x=7, color='r', label='mean = 7')
plt.axhline(y=1.2, color='r', label='std =1.2')
plt.title("Mean vs STD")
plt.xlabel("Mean")
plt.ylabel("Std")
plt.legend(loc='upper right')
plt.savefig('./Results/TCGA/meanvsstd.png')

## mean density plot - TCGA
plt.clf()
avgallTCGA.plot.density(label = 'All genes',bw_method=0.1, color='black')
plt.axvline(x=7, color='r', label='mean = 7')
plt.xlim(0,25) 
plt.xticks(np.arange(-1, 26, 2.5))
plt.title('Density Plot for the mean')
plt.legend(loc='upper right')
plt.xlim([0, 21])
plt.savefig('./Results/TCGA/meandensity.png', bbox_inches='tight')

## standard deviation density plot - TCGA
plt.clf()
stdallTCGA.plot.density(label = 'All genes',bw_method=0.1, color='black')
plt.axvline(x=1.2, color='r', label='std =1.2')
plt.xlim(0,6) 
plt.xticks(np.arange(0, 7,1))
plt.title('Density Plot for the standard deviation')
plt.legend(loc='upper right')
plt.xlim([0, 5.5])
plt.savefig('./Results/TCGA/stddensity.png', bbox_inches='tight')
