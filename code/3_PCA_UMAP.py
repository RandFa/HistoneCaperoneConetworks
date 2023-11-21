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


## filtering to get Highly variable genes (HVG)
dfGTExfiltered = dfGTExall.T.loc[(avgallGTEx>7) & (stdallGTEx>1.2)].T
dfTCGAfiltered = dfTCGAall.T.loc[(avgallTCGA>7) & (stdallTCGA>1.2)].T


## PCA and UMAP all genes - GTEx
##PCA
pcaGTEx = PCA()
gtexpca = pcaGTEx.fit_transform(StandardScaler().fit_transform(dfGTExfiltered))
exp_var_pca = pcaGTEx.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

##how many components explain 95 percent variance
list(np.array(cum_sum_eigenvalues) >= 0.95).index(True)

## visualization of percent of variance explained
plt.clf()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./Results/GTEx/pcacumulativeALL.png')

## visulaizing first and second PCs
plt.clf()
fig =px.scatter(x=gtexpca[:,0], y=gtexpca[:,1], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## visulaizing second and third and second PCs
plt.clf()
fig =px.scatter(x=gtexpca[:,1], y=gtexpca[:,2], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## UMAP 
gtexumap = umap.UMAP(min_dist = 0.5, spread= 1.5).fit_transform(dfGTExfiltered)

## visualization of first and second components of UNAP
fig =px.scatter(x=gtexumap[:,0], y=gtexumap[:,1], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()


## PCA and UMAP H/C - GTEx
##PCA
pcaGTEx = PCA()
gtexpca = pcaGTEx.fit_transform(StandardScaler().fit_transform(dfGTEx))
exp_var_pca = pcaGTEx.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

##how many components explain 95 percent variance
list(np.array(cum_sum_eigenvalues) >= 0.95).index(True)

## visualization of percent of variance explained
plt.clf()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./Results/GTEx/pcacumulativeHC.png')

## visulaizing first and second PCs
plt.clf()
fig =px.scatter(x=gtexpca[:,0], y=gtexpca[:,1], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## visulaizing second and third and second PCs
plt.clf()
fig =px.scatter(x=gtexpca[:,1], y=gtexpca[:,2], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## UMAP 
gtexumap = umap.UMAP(min_dist = 0.5, spread= 1.5).fit_transform(dfGTEx)

## visualization of first and second components of UNAP
fig =px.scatter(x=gtexumap[:,0], y=gtexumap[:,1], color=ygtex["spl_organ"])
fig.update_traces(marker={'size': 5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()



## PCA and UMAP all genes - TCGA
##PCA
pcaTCGA = PCA()
tcgapca = pcaTCGA.fit_transform(StandardScaler().fit_transform(dfTCGAfiltered))
exp_var_pca = pcaTCGA.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

##how many components explain 95 percent variance
list(np.array(cum_sum_eigenvalues) >= 0.95).index(True)

## visualization of percent of variance explained
plt.clf()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./Results/TCGA/pcacumulativeALL.png')

## visulaizing first and second PCs
plt.clf()
fig =px.scatter(x=tcgapca[:,0], y=tcgapca[:,1], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## visulaizing second and third and second PCs
plt.clf()
fig =px.scatter(x=tcgapca[:,1], y=tcgapca[:,2], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## UMAP 
tcgaumap = umap.UMAP(min_dist = 0.5, spread= 1.5).fit_transform(dfTCGAfiltered)

## visualization of first and second components of UNAP
fig =px.scatter(x=tcgaumap[:,0], y=tcgaumap[:,1], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()


## PCA and UMAP H/C - TCGA
##PCA
pcaTCGA = PCA()
tcgapca = pcaTCGA.fit_transform(StandardScaler().fit_transform(dfTCGA))
exp_var_pca = pcaTCGA.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

##how many components explain 95 percent variance
list(np.array(cum_sum_eigenvalues) >= 0.95).index(True)

## visualization of percent of variance explained
plt.clf()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./Results/TCGA/pcacumulativeHC.png')

## visulaizing first and second PCs
plt.clf()
fig =px.scatter(x=tcgapca[:,0], y=tcgapca[:,1], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## visulaizing second and third and second PCs
plt.clf()
fig =px.scatter(x=tcgapca[:,1], y=tcgapca[:,2], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 7.5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

## UMAP 
tcgaumap = umap.UMAP(min_dist = 0.5, spread= 1.5).fit_transform(dfTCGA)

## visualization of first and second components of UNAP
fig =px.scatter(x=tcgaumap[:,0], y=tcgaumap[:,1], color=ytcga["spl_organ"])
fig.update_traces(marker={'size': 5})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()
