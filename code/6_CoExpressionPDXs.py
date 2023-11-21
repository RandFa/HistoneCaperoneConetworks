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

##start of the analysis
## reading the expression dataframe 
dfpdx = pd.read_csv("./Data/pdxstmm.csv",index_col=0)

## reading meta data 
metapdx = pd.read_csv('./Data/pdxsmeta1.csv', index_col=0)
meta
## reading histones/chaperones gene list
hist = pd.read_csv('./Data/histone_chaperone.csv',index_col=0)


## filtering expression matrix for h/c gene only 
dfhist = pd.merge(hist[['GeneOfficialSymbol']], dfpdx,left_index=True, right_index=True)
dfhist.index = dfhist['GeneOfficialSymbol']
dfhist.drop('GeneOfficialSymbol', inplace=True, axis=1)

##fixinf column names 
dfhist.columns = [e[-3: ] for e in dfhist.columns]
dfpdx.columns = [e[-3: ] for e in dfpdx.columns]

##transposing the datasets
dftpdx = dfhist.T
dft2pdx = dfpdx.T

##renaming data columns
metapdx.columns = ["adryamicin_cyclophosphamide", "cisplatin_carboplatin", "capecitabine"]

##tidying the data
genes = dftpdx.columns.copy()
genegene = []
for g in genes:
    for gg in genes:
        genegene.append(g+ '-' + gg)

##scaling and centring
dfpdxs = pd.DataFrame(StandardScaler().fit_transform(dftpdx), index=dftpdx.index, columns=dftpdx.columns)

##merging the first treatment with meta
dfpdxs = pd.merge(dfpdxs, metapdx["adryamicin_cyclophosphamide"],left_index=True, right_index=True)


##empty dataframes and lists to store the results
intheat = pd.DataFrame(columns = genegene)
intpval = pd.DataFrame(columns = genegene)
intheatgene = pd.DataFrame(columns = genegene)
intpvalgene = pd.DataFrame(columns = genegene)
genelistc = []
genelistp = []
interlistc = []
interlistp = []

##GLM model for the first treatment
for g in genes:
        for gg in genes:
            if g < gg and dfpdxs[g].nunique() != 1 and dfpdxs[gg].nunique() != 1:
                model1 = glm("Q(g) ~ Q(gg) + Q(gg) : C(adryamicin_cyclophosphamide, Treatment(reference=0.0)) -1", dfpdxs)
                res = model1.fit()
                genelistc.append(res.params[0])
                genelistp.append(res.pvalues[0])
                interlistc.append(res.params[1])
                interlistp.append(res.pvalues[1])
            else:
                genelistc.append(0)
                genelistp.append(1)
                interlistc.append(0)
                interlistp.append(1)

##storing the coefficients and p values for the first treatment
intheat.loc["adryamicin_cyclophosphamide"] = interlistc
intpval.loc["adryamicin_cyclophosphamide"] = interlistp
intheatgene.loc["adryamicin_cyclophosphamide"] = genelistc
intpvalgene.loc["adryamicin_cyclophosphamide"] = genelistp


##merging the second treatment with meta
dfpdxs = pd.DataFrame(StandardScaler().fit_transform(dftpdx), index=dftpdx.index, columns=dftpdx.columns)
dfpdxs = pd.merge(dfpdxs, metapdx["cisplatin_carboplatin"],left_index=True, right_index=True)


##empty lists to store the values
genelistc = []
genelistp = []
interlistc = []
interlistp = []

##GLM model for the second treatment
for g in genes:
        for gg in genes:
            if g < gg and dfpdxs[g].nunique() != 1 and dfpdxs[gg].nunique() != 1:
                model1 = glm("Q(g) ~ Q(gg) + Q(gg) : C(cisplatin_carboplatin, Treatment(reference=0.0)) -1", dfpdxs)
                res = model1.fit()
                genelistc.append(res.params[0])
                genelistp.append(res.pvalues[0])
                interlistc.append(res.params[1])
                interlistp.append(res.pvalues[1])
            else:
                genelistc.append(0)
                genelistp.append(1)
                interlistc.append(0)
                interlistp.append(1)

##storing the coefficients and p values for the second treatment
intheat.loc["cisplatin_carboplatin"] = interlistc
intpval.loc["cisplatin_carboplatin"] = interlistp
intheatgene.loc["cisplatin_carboplatin"] = genelistc
intpvalgene.loc["cisplatin_carboplatin"] = genelistp

##merging the third treatment with meta
dfpdxs = pd.DataFrame(StandardScaler().fit_transform(dftpdx), index=dftpdx.index, columns=dftpdx.columns)
dfpdxs = pd.merge(dfpdxs, metapdx["capecitabine"],left_index=True, right_index=True)

##empty lists to store the values
genelistc = []
genelistp = []
interlistc = []
interlistp = []

##GLM model for the third treatment
for g in genes:
        for gg in genes:
            if g < gg and dfpdxs[g].nunique() != 1 and dfpdxs[gg].nunique() != 1:
                model1 = glm("Q(g) ~ Q(gg) + Q(gg) : C(capecitabine, Treatment(reference=0.0)) -1", dfpdxs)
                res = model1.fit()
                genelistc.append(res.params[0])
                genelistp.append(res.pvalues[0])
                interlistc.append(res.params[1])
                interlistp.append(res.pvalues[1])
            else:
                genelistc.append(0)
                genelistp.append(1)
                interlistc.append(0)
                interlistp.append(1)


##storing the coefficients and p values for the third treatment
intheat.loc["capecitabine"] = interlistc
intpval.loc["capecitabine"] = interlistp
intheatgene.loc["capecitabine"] = genelistc
intpvalgene.loc["capecitabine"] = genelistp

##saving coeffecients and their p values
intheat.to_csv('./Results/PDX/GLM/interactions.csv',encoding='utf-8', index=True)
intpval.to_csv('./Results/PDX/GLM/interactionspvalues.csv',encoding='utf-8', index=True)
intheatgene.to_csv('./Results/PDX/GLM/genes.csv',encoding='utf-8', index=True)
intpvalgene.to_csv('./Results/PDX/GLM/genespvalues.csv',encoding='utf-8', index=True)



##loading coeffecients and their p values if needed
coesT = pd.read_csv("./Results/PDX/GLM/interactions.csv",index_col=0)
pvalsT = pd.read_csv("./Results/PDX/GLM/interactionspvalues.csv",index_col=0)
coesG = pd.read_csv("./Results/PDX/GLM/genes.csv",index_col=0)
pvalsG = pd.read_csv("./Results/PDX/GLM/genespvalues.csv",index_col=0)

## tidying gene names
flist = ['H1-0', 'H1-7', 'H1-8', 'H1-10', 'H3-3A','H3-3B', 'H3-5']
newcol = []
for c in coesT.columns:
    added = False
    for ff in flist:
        if ff in c and added == False:
            fff = ff.replace('-','')
            newcol.append(c.replace(ff, fff))
            added = True
        elif ff in c and added == True:
            fff = ff.replace('-','')
            newcol.append(newcol[-1].replace(ff, fff))
            newcol.pop(-2)
            added = True   
    if added == False: 
        newcol.append(c)



## replacing gene names with the tidy ones
coesT.columns = newcol
pvalsT.columns = newcol
coesG.columns = newcol
pvalsG.columns = newcol


##removing genes against themselves
coesT =coesT[[i for i in coesT if len(set(coesT[i]))>1]]
pvalsT =pvalsT[[i for i in pvalsT if len(set(pvalsT[i]))>1]]
coesG =coesG[[i for i in coesG if len(set(coesG[i]))>1]]
pvalsG =pvalsG[[i for i in pvalsG if len(set(pvalsG[i]))>1]]

##filtering for significant coeffecients
coesT =  coesT.where(pvalsT <0.05)
coesG =  coesG.where(pvalsG <0.05)

##empty dataframe for stroing reults
finalres = pd.DataFrame(columns = ['treatment', 'gene_pairs', 'interaction_coeffe', 'gene_coeffec'])

##tidying column names of expression matrix
newcol = []
for c in dftpdx.columns:
    if c in flist:
        newcol.append(c.replace('-', ''))
    else:
        newcol.append(c)

dftpdx.columns = newcol
dfpdxs = pd.merge(dftpdx, metapdx,left_index=True, right_index=True)

##finding altered relationships and making scatterplot for them
for c in coesT.columns:
    res = list(coesT.index[coesT[c].notnull()])
    for pair in res:
        if np.isnan(coesT.loc[pair][c]) == False:
            if np.isnan(coesG.loc[pair][c]) == False:
                finalres.loc[len(finalres.index)] = [pair, c, coesT.loc[pair][c], coesG.loc[pair][c]]
                g1, g2 = c.split('-')
                dfhere1 = dfpdxs[dfpdxs[pair] == 0.0][[g1, g2]][dfpdxs[dfpdxs[pair] == 0.0][[g1, g2]].notnull().all(1)]
                dfhere2 = dfpdxs[dfpdxs[pair] == 1.0][[g1, g2]][dfpdxs[dfpdxs[pair] == 1.0][[g1, g2]].notnull().all(1)]
                if len(dfhere1.columns) != 0 and len(dfhere2.columns) != 0 : 
                    plt.clf()
                    plt.scatter(dfhere1[g1], dfhere1[g2], s = 30, c='black',label = "Resistant")
                    plt.scatter(dfhere2[g1], dfhere2[g2], s = 30, c='red',label ="Sensitive")
                    m1, b1 = np.polyfit(dfhere1[g1], dfhere1[g2], 1)
                    plt.plot(dfhere1[g1], m1*dfhere1[g1]+b1, color='black')
                    m2, b2 = np.polyfit(dfhere2[g1], dfhere2[g2], 1)
                    plt.plot(dfhere2[g1], m2*dfhere2[g1]+b2, color='red')
                    plt.title(g1 +' vs '+ g2)
                    plt.legend(loc='upper left')
                    plt.xlabel(g1)
                    plt.ylabel(g2)
                    plt.tight_layout()
                    plt.savefig('./Results/PDX/GLM/anticor/'+g1+g2+pair+'.png')         

##saving the final result matrix
finalres = finalres.drop_duplicates()
finalres.reset_index(inplace = True)
finalres.to_csv('./Results/PDX/GLM/anticor/finallist.csv',encoding='utf-8', index=True)


