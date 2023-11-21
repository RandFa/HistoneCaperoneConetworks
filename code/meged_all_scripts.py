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

### TMM normalization and log transformation
## loading count data 
df = pd.read_csv("./Data/GTExTCGAsmallcount.csv",index_col=0)

##normalizing by tmm
df_tmm = conorm.tmm(df)

## log transformation
df_log = np.log(df_tmm +0.1)

## saving the data 
df_log.to_csv('./Data/GTExTCGAsmalltmm.csv',encoding='utf-8', index=True)


##start of the analysis
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


##COEXPRESSION Relationships - GTEx
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
coes.to_csv('./Results/GTEx/GLM/coe.csv',encoding='utf-8', index=True)
pvals.to_csv('./Results/GTEx/GLM/pval.csv',encoding='utf-8', index=True)

##making a mask where pvalues are less than 0.05
pvalsfiltered = pvals.where(pvals<=0.05, np.nan) 

## heatmap of significant gene coeffecients
plt.clf()
plt.subplots(figsize=(20,15))
sns.heatmap(coes,  cmap= 'coolwarm', mask = pvalsfiltered.isnull(),center=0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM/heatmap.png')

## clustermap of significant gene coeffecients
plt.clf()
plt.subplots(figsize=(20,15))
sns.clustermap(coes,  cmap= 'coolwarm', mask = pvalsfiltered.isnull(),center=0,yticklabels=True,xticklabels=True)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM/clustermap.png')



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
            if g != gg:
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
coesT.to_csv('./Results/GTEx/GLM3/interactions.csv',encoding='utf-8', index=True)
pvalsT.to_csv('./Results/GTEx/GLM3/interactionspvalues.csv',encoding='utf-8', index=True)
coesG.to_csv('./Results/GTEx/GLM3/genes.csv',encoding='utf-8', index=True)
pvalsG.to_csv('./Results/GTEx/GLM3/genespvalues.csv',encoding='utf-8', index=True)

## heatmap of significant ineraction terms
coesT2 =coesT[[i for i in coesT if len(set(coesT[i]))>1]]
pvalsT2 =pvalsT[[i for i in pvalsT if len(set(coesT[i]))>1]]
pvalsfil = pvalsT2.where(pvalsT2 <=0.05, np.nan)
plt.clf()
fig, ax = plt.subplots(figsize=(300,250))
ax = sns.heatmap(coesT2,  cmap= 'coolwarm', center=0.0,mask = pvalsfil.isnull(), yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM3/heatinteractioncoeff.pdf')    

## heatmap of significant gene terms
coesG2 =coesG[[i for i in coesG if len(set(coesG[i]))>1]]
pvalsG2 =pvalsG[[i for i in pvalsG if len(set(coesG[i]))>1]]
pvalsfil = pvalsG2.where(pvalsG2 <=0.05, np.nan)
plt.clf()
fig, ax = plt.subplots(figsize=(300,250))
ax = sns.heatmap(coesG2,  cmap= 'coolwarm', center=0.0,mask = pvalsfil.isnull(), yticklabels=True,xticklabels=True, ax=ax)
ax.tick_params(labelsize=7)
# Saving the Seaborn Figure:
plt.savefig('./Results/GTEx/GLM3/heatGenecoeff.pdf')  


## Finding altered relationships GTEx
##loading data if needed
coesT = pd.read_csv("./Results/GTEx/GLM3/interactions.csv",index_col=0)
pvalsT = pd.read_csv("./Results/GTEx/GLM3/interactionspvalues.csv",index_col=0)
coesG = pd.read_csv("./Results/GTEx/GLM3/genes.csv",index_col=0)
pvalsG = pd.read_csv("./Results/GTEx/GLM3/genespvalues.csv",index_col=0)

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
        if abs(coesT.loc[pair][c]) >= 0.7:
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
                    plt.savefig('./Results/GTEx/GLM3/alteredrelations/'+g1+g2+rt+ot+'.png')


finalres = finalres.drop_duplicates()
finalres.reset_index(inplace = True)
finalres.to_csv('./Results/GTEx/GLM3/alteredrelations/alteredreltionships.csv',encoding='utf-8', index=True)

##INTEGRATING TCGA
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
genes = dfGTEx.columns.copy()
genegene = []
for g in genes:
    for gg in genes:
        genegene.append(g+ '-' + gg)


newcol = []
for c in dfGTEx.columns:
    if c in flist:
        newcol.append(c.replace('-', ''))
    else:
        newcol.append(c)


    
dfGTEx.columns = newcol
dfTCGA.columns = newcol

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
            if g != gg and dftglm[g].nunique() != 1 and dftglm[gg].nunique() != 1:
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
pvals11= intpval2.where(intpval2<=0.05, np.nan)

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
glm3gene1 = pd.read_csv("./Results/TCGA/GLM/genesgenes.csv",index_col=0)
glm3gpv1 = pd.read_csv("./Results/TCGA/GLM/genesgenespvalues.csv",index_col=0)

##tidying datframes
flist = ['H1-0', 'H1-7', 'H1-8', 'H1-10', 'H3-3A','H3-3B', 'H3-5']
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
        if np.isnan(glm3co1.loc[pair][c]) == False and abs(glm3co1.loc[pair][c])>=0.4:
            if np.isnan(glm3gene1.loc[pair][c]) == False and abs(glm3gene1.loc[pair][c])>=0.45:
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



##PDXs
##TMM for PDXs

## loading count data 
df = pd.read_csv("./Data/tablecounts_raw.csv",index_col=0)

##normalizing by tmm
df_tmm = conorm.tmm(df)

## log transformation
df_log = np.log(df_tmm +0.1)

## saving the data 
df_log.to_csv('./Data/pdxstmm.csv',encoding='utf-8', index=True)


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
intheatgene.to_csv('./Results/PDX/GLM/genesgenes.csv',encoding='utf-8', index=True)
intpvalgene.to_csv('./Results/PDX/GLM/genesgenespvalues.csv',encoding='utf-8', index=True)



##loading coeffecients and their p values if needed
coesT = pd.read_csv("./Results/PDX/GLM/interactions.csv",index_col=0)
pvalsT = pd.read_csv("./Results/PDX/GLM/interactionspvalues.csv",index_col=0)
coesG = pd.read_csv("./Results/PDX/GLM/genesgenes.csv",index_col=0)
pvalsG = pd.read_csv("./Results/PDX/GLM/genesgenespvalues.csv",index_col=0)

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
