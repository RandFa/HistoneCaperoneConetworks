## Importing required libraries
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
import anndata as ad

## Setting the working directory
os.chdir('./Dropbox (Institut Curie)/Sebastien Lemaire/Rand_Fatouh')


##loading cibersort results and meta dataframe
deconvoultionres = pd.read_csv('./Results/GTEx/cibersort/cibersort.csv', index_col=0)
meta = pd.read_csv('./Data/GTExTCGAsmallmeta.csv', index_col=0)

##splitting meta dataframe to get only gtex data 
ygtex = meta.loc[:5605,:][["spl_organ"]]
ygtex.set_index(deconvoultionres.index, inplace=True)

##merging deconvolution results with meta data
detis = pd.merge(deconvoultionres, ygtex,left_index=True, right_index=True)

##dropping unnecessary columns
detis.drop(['P-value', 'Correlation', 'RMSE'], axis=1, inplace=True)

##adding the index as a column
detis['sample'] = detis.index

## srting by the organ origin
detis = detis.sort_values(by=['spl_organ'])

## seting the index
detis.set_index(['spl_organ', 'sample'], inplace= True)

##making a heatmap of the results
detis.head()
plt.clf()
sns.set(font_scale=7)
plt.subplots(figsize=(210,210))
sns.heatmap(detis,  cmap="crest",yticklabels=80,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/cibersort/heatmap.png')
