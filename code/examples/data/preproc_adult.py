import numpy as np
import pandas as pd
import io, os, requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import itertools
import pickle as pk

def load_dataset(path, urls):
  if not os.path.exists(path):
    os.mkdir(path)
  for url in urls:
    data = requests.get(url).content
    filename = os.path.join(path, os.path.basename(url))
    with open(filename, "wb") as file:
      file.write(data)
  return

def demographic_groups(df, cap=50):
  ages = [(0,25), (25,30), (30,35), (35,40), (40,45), (45,55), (55,max(df['age']))]
  race = set(df['race']); race.remove('Other')
  gender = set(df['sex'])
  groups=[]
  for (a,r,g) in itertools.product(ages, race, gender):
    ng = df.index[(df['race']==r)&(df['sex']==g)&(a[0]<df['age'])&(df['age']<=a[1])].tolist()
    groups+=[[df.index.get_loc(n) for n in ng[:cap]]]
  #save results
  f = open('groups_sensemake_adult.pk', 'wb')
  pk.dump( (groups,list(itertools.product(ages, race, gender))), f)
  f.close()
  return

def vq_demographic_groups(df, cap=50):
  ages = [(0,25), (25,30), (30,35), (35,40), (40,45), (45,55), (55,max(df['age']))]
  race = set(df['race'])
  race = set(df['race']); race.remove('Other')
  gender = set(df['sex'])
  groups=[]
  quality = [0,1,2]
  for (q,a,r,g) in itertools.product(quality, ages, race, gender):
    ng = df.index[(df['race']==r)&(df['sex']==g)&(a[0]<df['age'])&(df['age']<=a[1])].tolist()
    print(len(ng), a, r, g)
    if len(ng)>=3*cap:
      groups+=[[df.index.get_loc(n) for n in ng[q*cap:(q+1)*cap]]]
    else:
      groups+=[[df.index.get_loc(n) for n in ng[int(q*float(len(ng))/3.):int((q+1)*float(len(ng))/3.)]]]
  #save results
  f = open('vq_groups_sensemake_adult.pk', 'wb')
  pk.dump( (groups,list(itertools.product(quality, ages, race, gender))), f)
  f.close()
  return


urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
load_dataset('data', urls)

columns = ["age", "workClass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
          "native-country", "income"]
train_data = pd.read_csv('data/adult.data', names=columns,
             sep=' *, *', na_values='?', engine="python").dropna()
test_data  = pd.read_csv('data/adult.test', names=columns,
             sep=' *, *', skiprows=1, na_values='?', engine="python").dropna()

X, Xt = train_data[columns[::-1]], test_data[columns[::-1]]
y = [-1 if s=='<=50K' else 1 for s in train_data["income"]]
yt = [-1 if s=='<=50K.' else 1 for s in test_data["income"]]

demographic_groups(X)
vq_demographic_groups(X)

# numerical columns : standardize
numcols = ['age', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
ss=StandardScaler()
ss.fit(X[numcols])
Xnum, Xtnum = ss.transform(X[numcols]), ss.transform(Xt[numcols])

# categorical columns: apply 1-hot-encoding
catcols = ['workClass', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country']
enc = OneHotEncoder()
enc.fit(X[catcols])
Xcat, Xtcat = enc.transform(X[catcols]).toarray(), enc.transform(Xt[catcols]).toarray()
X, Xt = np.concatenate((Xnum, Xcat), axis=1), np.concatenate((Xtnum, Xtcat), axis=1)

pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
Xt = pca.transform(Xt)
X = np.c_[ X, np.ones(X.shape[0])]
Xt = np.c_[ Xt, np.ones(Xt.shape[0])]

np.savez('adult', X=X, y=y, Xt=Xt, yt=yt)
