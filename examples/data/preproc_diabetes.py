import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import itertools
import pickle as pk

def demographic_groups(df):
  ages = set(df['age'])
  race = set(df['race'])
  gender = set(df['gender'])
  groups, idcs = [], []
  idx = 0
  for idx, (a,r,g) in enumerate(itertools.product(ages, race, gender)):
    ng = df.index[(df['race']==r)&(df['gender']==g)&(df['age']==a)].tolist()
    if len(ng)>40:
      groups+=[[df.index.get_loc(n) for n in ng]]
      idcs+=[idx]
  #save results
  f = open('groups_sensemake_diabetes.pk', 'wb')
  pk.dump( (groups,[list(itertools.product(ages, race, gender))[idx] for idx in idcs]), f)
  f.close()
  return


def vq_demographic_groups(df, cap=100):
  ages = set(df['age'])
  race = set(df['race'])
  gender = set(df['gender'])
  groups, idcs = [], []
  idx = 0
  quality = [0,1,2]
  for idx, (q, a,r,g) in enumerate(itertools.product(quality, ages, race, gender)):
    ng = df.index[(df['race']==r)&(df['gender']==g)&(df['age']==a)].tolist()
    if len(ng)>40:
      if len(ng)>=3*cap:
        groups+=[[df.index.get_loc(n) for n in ng[q*cap:(q+1)*cap]]]
      else:
        groups+=[[df.index.get_loc(n) for n in ng[int(q*float(len(ng))/3.):int((q+1)*float(len(ng))/3.)]]]
      idcs+=[idx]
  #save results
  f = open('vq_groups_sensemake_diabetes.pk', 'wb')
  pk.dump( (groups,[list(itertools.product(quality, ages, race, gender))[idx] for idx in idcs]), f)
  f.close()
  return


columns = ["race" , "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital",
          "num_lab_procedures", "num_procedures",	"num_medications", "number_outpatient",	"number_emergency",	"number_inpatient",
          "diag_1",	"diag_2",	"diag_3",	"number_diagnoses", "A1Cresult",	"metformin", "repaglinide",	"nateglinide","chlorpropamide",
          "glimepiride",	"acetohexamide",	"glipizide",	"glyburide", "tolbutamide",	"pioglitazone", "rosiglitazone",	"acarbose",
          "miglitol",	"troglitazone",	"tolazamide",	"examide",	"citoglipton", "insulin",	"glyburide-metformin",	"glipizide-metformin",
          "glimepiride-pioglitazone",	"metformin-rosiglitazone", "metformin-pioglitazone", "change",	"diabetesMed", "readmitted"]

data = pd.read_csv('diabetes.csv',  na_values='?')
# consider the first encounter of each patient
data = data.sort_values('encounter_id').groupby('patient_nbr').first()
data = data.reset_index()
data = data[columns].dropna()
data.dropna(inplace=True)
data = data[columns[::-1]]
y = [-1 if s=='NO' else 1 for s in data["readmitted"]]

# numerical columns : standardize
numcols = ["time_in_hospital", "num_lab_procedures", "num_procedures",	"num_medications", "number_outpatient",	"number_emergency",	"number_inpatient",
          "diag_1",	"diag_2",	"diag_3",	"number_diagnoses"]
catcols = [c for c in columns if c not in numcols]
data[numcols] = data[numcols].apply(pd.to_numeric, errors='coerce')
data[catcols] = data[catcols].astype(str)
szd = 60000
N=data.shape[0]
X, Xt = data.head(n=szd), data.tail(n=N-szd)
idxX = list(X.index)
idxXt = list(Xt.index)
y, yt = y[:szd], y[-(N-szd):]
ss=StandardScaler()
ss.fit(X[numcols])
Xnum, Xtnum = ss.transform(X[numcols]), ss.transform(Xt[numcols])

# categorical columns: apply 1-hot-encoding
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[catcols])
Xcat, Xtcat = enc.transform(X[catcols]).toarray(), enc.transform(Xt[catcols]).toarray()
X, Xt = np.concatenate((Xnum, Xcat), axis=1), np.concatenate((Xtnum, Xtcat), axis=1)

idxXnotnan = np.argwhere(~np.isnan(X).any(axis=1)).flatten()
idxXtnotnan = np.argwhere(~np.isnan(Xt).any(axis=1)).flatten()
onlyInX = [idxX[e] for e in idxXnotnan]
onlyInXt = [idxXt[e] for e in idxXtnotnan]
dataX = data.iloc[onlyInX]
X, y = X[list(idxXnotnan),:], [y[e] for e in list(idxXnotnan)]
Xt, yt = Xt[list(idxXtnotnan),:], [yt[e] for e in list(idxXtnotnan)]

print(X.shape, Xt.shape, X[:10,:], Xt[:10,:])
pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
Xt = pca.transform(Xt)
X = np.c_[ X, np.ones(X.shape[0])]
Xt = np.c_[ Xt, np.ones(Xt.shape[0])]

np.savez('diabetes', X=X, y=y, Xt=Xt, yt=yt)
demographic_groups(dataX)
vq_demographic_groups(dataX)
