# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:36:11 2020

@author: Sima Soltani
"""

import pandas as pd

df = pd.DataFrame([['green','M',10.1,'class2'],
                   ['red','L',13.5,'class1'],
                   ['blue','XL',15.3,'class2']])
df.columns=['color','size','price','classlabel']


size_mapping = {'XL':3,
                'L':2,
                'M':1}
df['size']=df['size'].map(size_mapping)
inv_Size_mapping = {v:k for k, v in size_mapping.items()}


import numpy as np

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}

df['classlabel']=df['classlabel'].map(class_mapping)

inv_class_mapping = {v:k for k,v in class_mapping.items() }
df['classlabel']=df['classlabel'].map(inv_class_mapping)

#labelencoder sklearn
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

#inversing to get the original
y2 = class_le.inverse_transform(y)


#one-hot encoding
from sklearn.preprocessing import OneHotEncoder
X=df[['color','size','price']].values
color_ohe=OneHotEncoder(categories='auto',drop='first')
color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray()
#dropping one of the columns 

#ColumnTransformer
from sklearn.compose import ColumnTransformer
c_trans = ColumnTransformer(
    [('onehot',OneHotEncoder(categories='auto',drop='first'),[0]),
     ('nothing','passthrough',[1,2])
     ])
c_trans.fit_transform(X).astype(float)

#more convinient way :get_dummies
pd.get_dummies(df[['price','size','color']],drop_first=True)



import pandas as pd
import numpy as np
df_wine=pd.read_csv('https://archive.ics.uci.edu/'
                    'ml/machine-learning-databases/'
                    'wine/wine.data',header=None)


df_wine.columns = ['Class label','Alcohol',
                   'Malic Acid','Ash',
                   'Alcalinity of ash','Magnesium',
                   'Total Phenoles','Flavanoids',
                   'Nonflaveoid phenold',
                   'Prantocyanins',
                   'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

np.unique(df_wine['Class label'])


from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
from sklearn.preprocessing import StandardScaler
stdscr = StandardScaler()
X_train_std = stdscr.fit_transform(X_train)
X_test_std = stdscr.transform(X_test)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1',
                        C=1.0,
                        solver = 'liblinear')


lr.fit(X_train_std,y_train)
lr.score(X_train_std,y_train)
lr.score(X_test_std,y_test)
lr.intercept_
lr.coef_


##testing different strength of regularizations effect on weights- with plot
import matplotlib.pyplot as plt
fig = plt.figure()
ax =plt.subplot(111)

colors = ['blue','green','red','cyan',
          'magenta','yellow','black',
          'pink','lightgreen','lightblue',
          'gray','indigo','orange']
weights,params=[],[]
for c in np.arange(-4.0,6.0):
    lr = LogisticRegression(penalty='l1',
                            C=10.**c,
                            solver='liblinear',
                            multi_class='ovr',random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    
weights=np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],
             label=df_wine.columns[column+1],
             color=color)
plt.axhline(0,color='black',linestyle='--',linewidth =3)
plt.xlim([10**(-5),10**5])
plt.ylabel('Weight coefficeint')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor =(1.38,1.03),
          ncol=1,fancybox = True)
plt.show()


#feature selection
from SBS import SBS

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker = 'o')
plt.ylim([0.7,1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# test our feature seection
k3 = list (sbs.subsets_[10])
knn.fit(X_train_std[:,k3],y_train)
knn.score(X_train_std[:,k3],y_train)
knn.score(X_test_std[:,k3],y_test)

# using random forest for feature selection
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]
forest=RandomForestClassifier(n_estimators=500,
                               random_state=1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f +1,30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title ('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align = 'center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation = 90)

plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()
