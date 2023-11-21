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

## filtering for tissues in both data sets
indexok = list(meta[~meta['spl_organ'].isin(['BLCA', 'HNSC', 'MESO', 'PCPG', 'READ', 'THYM', 'UVM'])].index) 
meta = pd.read_csv('./Data/GTExTCGAsmallmeta2.csv', index_col=0)
dft2 = dft2.iloc[indexok,:]
dft = dft.iloc[indexok,:]
dfTCGA
##splitting gtex and tcga - h/c and the whole dataset (all)
dfGTEx = dft.iloc[:5604,:]
dfGTExall = dft2.iloc[:5605,:]
dfTCGA = dft.iloc[5604:,:]
dfTCGAall = dft2.iloc[5605:,:]
ygtex = meta.loc[:5604,:][["spl_organU"]]
ytcga = meta.loc[5605:,:][["spl_organU"]]
ygtex.set_index(dfGTEx.index, inplace=True)
ytcga.set_index(dfTCGA.index, inplace=True)
dfGTEx.columns = hist['GeneOfficialSymbol'].loc[dfGTEx.columns]
dfTCGA.columns = hist['GeneOfficialSymbol'].loc[dfTCGA.columns]

##tidyung gene names
flist = ['H1-0', 'H1-7', 'H1-8', 'H1-10', 'H3-3A','H3-3B', 'H3-5']
newcol = []
for c in dfGTEx.columns:
    if c in flist:
        newcol.append(c.replace('-', ''))
    else:
        newcol.append(c)


dfGTEx.columns = newcol
dfTCGA.columns = newcol
genes = dfGTEx.columns.copy()[:-1]
genegene = []
for g in genes:
    for gg in genes:
        genegene.append(g+ '-' + gg)




##empty dataframes to store values
intheat = pd.DataFrame(columns = genegene)
intpval = pd.DataFrame(columns = genegene)
intheatgene = pd.DataFrame(columns = genegene)
intpvalgene = pd.DataFrame(columns = genegene)
tissues = sorted(list(ytcga['spl_organU'].unique()))
tissues.remove('Artery')

## scaling and centring each dataset per tissue and merging with the organ class
dfGTEx= pd.merge(dfGTEx, ygtex['spl_organU'],left_index=True, right_index=True)
dfscaled = dfGTEx.groupby('spl_organU').transform(lambda x: StandardScaler().fit_transform(x.values[:,np.newaxis]).ravel())
gtxglm = pd.merge(dfscaled, ygtex,left_index=True, right_index=True)
dfTCGA=pd.merge(dfTCGA, ytcga['spl_organU'],left_index=True, right_index=True)
dfscaledt = dfTCGA.groupby('spl_organU').transform(lambda x: StandardScaler().fit_transform(x.values[:,np.newaxis]).ravel())
tcgaglm = pd.merge(dfscaledt, ytcga,left_index=True, right_index=True)


##GLM Model
for t in range(len(tissues)):
    genelistc = []
    genelistp = []
    interlistc = []
    interlistp = []
    dftgtx = gtxglm[gtxglm['spl_organU']== tissues[t]]
    dftgtx['dataset'] = 'GTEx'
    dfttcga = tcgaglm[tcgaglm['spl_organU']== tissues[t]]
    dfttcga['dataset'] = 'TCGA'
    dftglm = pd.concat([dftgtx, dfttcga])
    for g in genes:
        for gg in genes:
            print(g, gg)
            if g < gg and dftglm[g].nunique() != 1 and dftglm[gg].nunique() != 1:
                model1 = glm("Q(g) ~ Q(gg) + Q(gg) : C(dataset, Treatment(reference='GTEx')) -1", dftglm)
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
            print(len(genelistc))
    intheat.loc[tissues[t]] = interlistc
    intpval.loc[tissues[t]] = interlistp
    intheatgene.loc[tissues[t]] = genelistc
    intpvalgene.loc[tissues[t]] = genelistp




##saving interaction terms and their p values
intheat.to_csv('./Results/TCGA/GLM/interactions.csv',encoding='utf-8', index=True)
intpval.to_csv('./Results/TCGA/GLM/interactionspvalues.csv',encoding='utf-8', index=True)

##filtering for heatmap
intheat2 =intheat[[i for i in intheat if len(set(intheat[i]))>1]]
intpval2 =intpval[[i for i in intheat if len(set(intheat[i]))>1]]
pvals11= intpval2.where(intpval2<0.05, np.nan)

#heatmap of interaction coeffecients
plt.clf()
fig, ax = plt.subplots(figsize=(300,250))
ax = sns.heatmap(intheat2,  cmap= 'coolwarm', center=0.0,mask = pvals11.isnull(), yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/TCGA/GLM/heatinteractioncoeff.pdf')    

##saving gene coeffecients and their p values
intheatgene.to_csv('./Results/TCGA/GLM/genesgenes.csv',encoding='utf-8', index=True)
intpvalgene.to_csv('./Results/TCGA/GLM/genesgenespvalues.csv',encoding='utf-8', index=True)

## filtering the values for heatmap
intheatgene2 =intheatgene[[i for i in intheatgene if len(set(intheatgene[i]))>1]]
intpvalgene2 =intpvalgene[[i for i in intheatgene if len(set(intheatgene[i]))>1]]
pvals11= intpvalgene2.where(intpvalgene2<=0.05, np.nan)

##heatmap of gene coeffecients
plt.clf() 
fig, ax = plt.subplots(figsize=(250,75))
ax = sns.heatmap(intheatgene2 , cmap= 'coolwarm', mask = pvals11.isnull(), center=0.0,yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/TCGA/GLM/heatgenecoeff.pdf')

##Looking for altered relationships
##Loading data
glm3co1 = pd.read_csv("./Results/TCGA/GLM/interactions.csv",index_col=0)
glm3pv1 = pd.read_csv("./Results/TCGA/GLM/interactionspvalues.csv",index_col=0)
glm3gene1 = pd.read_csv("./Results/TCGA/GLM/genes.csv",index_col=0)
glm3gpv1 = pd.read_csv("./Results/TCGA/GLM/genespvalues.csv",index_col=0)

##tidying datframes
newcol = []
for c in glm3co1.columns:
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

##tidying the data

glm3co1.columns = newcol
glm3pv1.columns = newcol
glm3gene1.columns = newcol
glm3gpv1.columns = newcol
glm3co1 =glm3co1[[i for i in glm3co1 if len(set(glm3co1[i]))>1]]
glm3pv1 =glm3pv1[[i for i in glm3pv1 if len(set(glm3pv1[i]))>1]]
glm3gene1 =glm3gene1[[i for i in glm3gene1 if len(set(glm3gene1[i]))>1]]
glm3gpv1 =glm3gpv1[[i for i in glm3gpv1 if len(set(glm3gpv1[i]))>1]]

##filtering for insignificant pvalues
glm3co1 =  glm3co1.where(glm3pv1 <0.05)
glm3gene1 =  glm3gene1.where(glm3gpv1 <0.05)

##empty datafame for final results
finalres = pd.DataFrame(columns = ['tissue_pairs', 'gene_pairs', 'interaction_coeffe', 'gene_coeffec'])


##Looking for altered relationships
for c in glm3co1.columns:
    res = list(glm3co1.index[glm3co1[c].notnull()])
    for pair in res:
        if np.isnan(glm3co1.loc[pair][c]) == False and abs(glm3co1.loc[pair][c])>=0.7:
            if np.isnan(glm3gene1.loc[pair][c]) == False:
                finalres.loc[len(finalres.index)] = [pair, c, glm3co1.loc[pair][c], glm3gene1.loc[pair][c]]
                g1, g2 = c.split('-')
                plt.clf()
                plt.scatter(dfTCGA[dfTCGA.spl_organU == pair][g1], dfTCGA[dfTCGA.spl_organU == pair][g2], s = 5, c='red',label = 'Cancer')
                plt.scatter(dfGTEx[dfGTEx.spl_organU == pair][g1], dfGTEx[dfGTEx.spl_organU == pair][g2], s = 5, c='black',label = 'Healthy')
                m1, b1 = np.polyfit(dfGTEx[dfGTEx.spl_organU == pair][g1], dfGTEx[dfGTEx.spl_organU == pair][g2], 1)
                plt.plot(dfGTEx[dfGTEx.spl_organU == pair][g1], m1*dfGTEx[dfGTEx.spl_organU == pair][g1]+b1, color='black')
                m2, b2 = np.polyfit(dfTCGA[dfTCGA.spl_organU == pair][g1], dfTCGA[dfTCGA.spl_organU == pair][g2], 1)
                plt.plot(dfTCGA[dfTCGA.spl_organU == pair][g1], m2*dfTCGA[dfTCGA.spl_organU == pair][g1]+b2, color='red')
                plt.title(g1 +' vs '+ g2+ ' in '+ pair)
                plt.legend(loc='upper left')
                plt.xlabel(g1)
                plt.ylabel(g2)
                plt.tight_layout()
                plt.savefig('./Results/TCGA/GLM/anticor/'+g1+g2+pair+'.png')


##saving final altered relationships
finalres.to_csv('./Results/TCGA/GLM/anticor/finallist.csv',encoding='utf-8', index=True)

