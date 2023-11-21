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
df = pd.read_csv("./Data/GTExTCGAsmalltmm.csv",index_col=0)

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
ygtex = meta.loc[:5605,:][["spl_organ"]]
ytcga = meta.loc[5605:,:][["spl_organ"]]

dfGTEx.columns = hist['GeneOfficialSymbol'].loc[dfGTEx.columns]
dfTCGA.columns = hist['GeneOfficialSymbol'].loc[dfTCGA.columns]

###Pearson correlation
genes = dfGTEx.columns.copy()
coesp = pd.DataFrame(index=genes)
pvalsp = pd.DataFrame(index=genes)
for g in genes:
    coe = []
    pval = [] 
    for i in genes:
        coef, p = stats.pearsonr(dfGTEx[g],dfGTEx[i])
        coe.append(coef) 
        pval.append(p)
    coesp[g] = coe
    pvalsp[g] = pval


## saving correlation values and p vlaues
coesp.to_csv('./Results/GTEx/correlation/corr.csv',encoding='utf-8', index=True)
pvalsp.to_csv('./Results/GTEx/correlation/pval.csv',encoding='utf-8', index=True)

##making a lmask for insignifcant values
pvals11 = pvalsp.where(pvalsp<=0.05, np.nan) 
##Clustermap
plt.clf()
plt.subplots(figsize=(20,15))
sns.clustermap(coesp,  cmap= 'coolwarm', mask = pvals11.isnull(),center=0.0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/correlation/clustermap.png')

##Heatmap
plt.clf()
plt.subplots(figsize=(20,15))
sns.heatmap(coesp,  cmap= 'coolwarm', mask = pvals11.isnull(),center=0.0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/correlation/heatmap.png')



## GLM simple model - GTEx
## setting up the dataframe
tissues = sorted(list(ygtex['spl_organ'].unique()))
genes = dfGTEx.columns.copy()
ygtex.set_index(dfGTEx.index, inplace=True)

#scaling and centring 
dfscaled = pd.DataFrame(StandardScaler().fit_transform(dfGTEx), index=dfGTEx.index, columns=dfGTEx.columns)

## merging with tissue type
gtxglm = pd.merge(dfscaled, ygtex,left_index=True, right_index=True)

## dataframes for storing coeffecients and pvalues
coes = pd.DataFrame(index=genes)
pvals = pd.DataFrame(index=genes)

## GLM for each possible pairs
for g in genes:
    coe = []
    pval = []
    for i in genes:
        if g == i:
            coe.append(1.0)
            pval.append(1.0)
        else:           
            model1 = glm("Q(g)  ~  Q(i)", gtxglm)
            res = model1.fit()
            coe.append(res.params[1])
            pval.append(res.pvalues[1])
    coes[g] = coe
    pvals[g] = pval

##storing the coeffecients and pvalues to csv
coes.to_csv('./Results/GTEx/GLM0/coe.csv',encoding='utf-8', index=True)
pvals.to_csv('./Results/GTEx/GLM0/pval.csv',encoding='utf-8', index=True)

##making a mask where pvalues are less than 0.05
pvalsfiltered = pvals.where(pvals<=0.05, np.nan) 

## heatmap of significant gene coeffecients
plt.clf()
plt.subplots(figsize=(20,15))
sns.heatmap(coes,  cmap= 'coolwarm', mask = pvalsfiltered.isnull(),center=0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM0/heatmap.png')

## clustermap of significant gene coeffecients
plt.clf()
plt.subplots(figsize=(20,15))
sns.clustermap(coes,  cmap= 'coolwarm', mask = pvalsfiltered.isnull(),center=0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM0/clustermap.png')



##GLM with tissue interaction terms
#scaling and centring per tissue
dfGTEx = pd.merge(dfGTEx, ygtex['spl_organ'],left_index=True, right_index=True)
dfscaled = dfGTEx.groupby('spl_organ').transform(lambda x: StandardScaler().fit_transform(x.values[:,np.newaxis]).ravel())
dfscaled = pd.merge(dfscaled, ygtex['spl_organ'],left_index=True, right_index=True)

##tidying gene names
genegene = []
for g in genes:
    for gg in genes:
        genegene.append(g+ '-' + gg)

## empty datframes for sttoring gene, interacton coeffecients and thier p values
coesT = pd.DataFrame(columns = genegene)
pvalsT = pd.DataFrame(columns = genegene)
coesG = pd.DataFrame(columns = genegene)
pvalsG = pd.DataFrame(columns = genegene)

##GLM model
for t in range(len(tissues)):
    thistiss = tissues[:t]+tissues[t+1:]
    thistiss = [tissues[t] + '-' + s for s in thistiss]
    intheat2 = pd.DataFrame(index=thistiss)
    intpval2 = pd.DataFrame(index=thistiss)
    intheatgene2 = pd.DataFrame(index=[tissues[t]])
    intpvalgene2 = pd.DataFrame(index=[tissues[t]])
    for g in genes:
        for gg in genes:
            if g < gg:
                model1 = glm("Q(g) ~ Q(gg) * C(spl_organ, Treatment(reference=tissues[t]))", dfscaled)
                res = model1.fit()
                intheat2[g+ '-' + gg] = list(res.params[len(tissues)+1:])
                intpval2[g+ '-' + gg] = list(res.pvalues[len(tissues)+1:])
                intheatgene2[g+ '-' + gg] = res.params[len(tissues)]
                intpvalgene2[g+ '-' + gg] = res.pvalues[len(tissues)]
            else:
                intheat2[g+ '-' + gg] = [0.0] * 28 
                intpval2[g+ '-' + gg] = [1.0] * 28
                intheatgene2[g+ '-' + gg] = 0.0
                intpvalgene2[g+ '-' + gg] = 1.0
    coesT =pd.concat([coesT, intheat2])
    pvalsT= pd.concat([pvalsT, intpval2])
    coesG = pd.concat([coesG, intheatgene2])
    pvalsG = pd.concat([pvalsG, intpvalgene2])

## saving the datframes
coesT.to_csv('./Results/GTEx/GLM/interactions.csv',encoding='utf-8', index=True)
pvalsT.to_csv('./Results/GTEx/GLM/interactionspvalues.csv',encoding='utf-8', index=True)
coesG.to_csv('./Results/GTEx/GLM/genes.csv',encoding='utf-8', index=True)
pvalsG.to_csv('./Results/GTEx/GLM/genespvalues.csv',encoding='utf-8', index=True)

## heatmap of significant ineraction terms
coesT2 =coesT[[i for i in coesT if len(set(coesT[i]))>1]]
pvalsT2 =pvalsT[[i for i in pvalsT if len(set(coesT[i]))>1]]
pvalsfil = pvalsT2.where(pvalsT2 <=0.05, np.nan)
plt.clf()
fig, ax = plt.subplots(figsize=(300,250))
ax = sns.heatmap(coesT2,  cmap= 'coolwarm', center=0.0,mask = pvalsfil.isnull(), yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM/heatinteractioncoeff.pdf')    

## heatmap of significant gene terms
coesG2 =coesG[[i for i in coesG if len(set(coesG[i]))>1]]
pvalsG2 =pvalsG[[i for i in pvalsG if len(set(coesG[i]))>1]]
pvalsfil = pvalsG2.where(pvalsG2 <=0.05, np.nan)
plt.clf()
fig, ax = plt.subplots(figsize=(300,250))
ax = sns.heatmap(coesG2,  cmap= 'coolwarm', center=0.0,mask = pvalsfil.isnull(), yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM/heatGenecoeff.pdf')  


## Finding altered relationships GTEx
##loading data if needed
coesT = pd.read_csv("./Results/GTEx/GLM/interactions.csv",index_col=0)
pvalsT = pd.read_csv("./Results/GTEx/GLM/interactionspvalues.csv",index_col=0)
coesG = pd.read_csv("./Results/GTEx/GLM/genes.csv",index_col=0)
pvalsG = pd.read_csv("./Results/GTEx/GLM/genespvalues.csv",index_col=0)

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
finalres = pd.DataFrame(columns = ['tissue_pairs', 'gene_pairs', 'interaction_coeffe', 'gene_coeffec'])

##tidying column names of expression matrix
newcol = []
for c in dfGTEx.columns:
    if c in flist:
        newcol.append(c.replace('-', ''))
    else:
        newcol.append(c)

    
dfGTEx.columns = newcol

##finding altered relationships and making scatterplot for them
for c in coesT.columns:
    res = list(coesT.index[coesT[c].notnull()])
    for pair in res:
        rt, ot = pair.split('-')
        if abs(coesT.loc[pair][c]) >= 0.75:
            if np.isnan(coesG.loc[rt][c]) == False:
                finalres.loc[len(finalres.index)] = [pair, c, coesT.loc[pair][c], coesG.loc[rt][c]]
                g1, g2 = c.split('-')
                dfhere1 = dfGTEx[dfGTEx.spl_organ == rt][[g1, g2]][dfGTEx[dfGTEx.spl_organ == rt][[g1, g2]].notnull().all(1)]
                dfhere2 = dfGTEx[dfGTEx.spl_organ == ot][[g1, g2]][dfGTEx[dfGTEx.spl_organ == ot][[g1, g2]].notnull().all(1)]
                if len(dfhere1.columns) != 0 and len(dfhere2.columns) != 0 : 
                    plt.clf()
                    plt.scatter(dfhere1[g1], dfhere1[g2], s = 30, c='black',label = rt)
                    plt.scatter(dfhere2[g1], dfhere2[g2], s = 30, c='red',label =ot)
                    m1, b1 = np.polyfit(dfhere1[g1], dfhere1[g2], 1)
                    plt.plot(dfhere1[g1], m1*dfhere1[g1]+b1, color='black')
                    m2, b2 = np.polyfit(dfhere2[g1], dfhere2[g2], 1)
                    plt.plot(dfhere2[g1], m2*dfhere2[g1]+b2, color='red')
                    plt.title(g1 +' vs '+ g2)
                    plt.legend(loc='upper left')
                    plt.xlabel(g1)
                    plt.ylabel(g2)
                    plt.tight_layout()
                    plt.savefig('./Results/GTEx/GLM/anticor/'+g1+g2+rt+ot+'.png')


##saving the final result matrix
finalres = finalres.drop_duplicates()
finalres.reset_index(inplace = True)
finalres.to_csv('./Results/GTEx/GLM3/anticor/finallist.csv',encoding='utf-8', index=True)
