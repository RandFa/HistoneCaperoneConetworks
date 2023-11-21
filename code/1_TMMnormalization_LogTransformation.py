## importing libraries
import os
import pandas as pd
import matplotlib
import numpy as np
import conorm

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



##TMM for PDXs

## loading count data 
df = pd.read_csv("./Data/tablecounts_raw.csv",index_col=0)

##normalizing by tmm
df_tmm = conorm.tmm(df)

## log transformation
df_log = np.log(df_tmm +0.1)

## saving the data 
df_log.to_csv('./Data/pdxstmm.csv',encoding='utf-8', index=True)











