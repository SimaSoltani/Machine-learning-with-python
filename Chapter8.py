# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:41:16 2020

@author: Sima Soltani
"""
#obtaining data set

import tarfile

with tarfile.open('data\\aclImdb_v1.tar.gz','r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)

# preprocessing the data

import pyprind
import pandas as pd
import os

#change the 'basepath' to the directory of the unzipped movie dataset

basepath = 'aclImdb'

labels ={'pos':1,'neg':0}
pbar =pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),
                      'r',encoding = 'utf-8') as infile:
                txt=infile.read()
                
            df = df.append([[txt,labels[l]]],
                           ignore_index = True)
            pbar.update()
df.columns = ['review','sentiment']

import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index = False, encoding = 'utf-8')

df = pd.read_csv('movie_data.csv',encoding = 'utf-8')
df.head(3)
df.shape

#transforming word into feature vectors

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet,'
                 'and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
bag.toarray()

#
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf = True)
np.set_printoptions(precision = 2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#regular expression to remove unwanted characters
import re
def preprocessor(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = (re.sub('[\W]+', ' ' ,text.lower())+' '.join(emoticons).replace('-',''))
    return text

preprocessor(df.loc[0,'review'][-50:])
preprocessor("</a>This :) is :( a test:-)!")    

df['review']=df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()
import nltk
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
nltk.download('stopwords')
# after we download the stop-words set we can load and apply the English 
# stop-word set as follows:

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes '
                             'running and runs a lot')[-10:] if w not in stop]

#logistic regression for classification of the reviews
X_train = df.loc[:25000,'review'].values
X_test = df.loc[25000:,'review'].values
y_train = df.loc[:25000,'sentiment'].values
y_test = df.loc[25000:,'sentiment'].values


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                         lowercase = False,
                         preprocessor = None)
param_grid = [{'vect__ngram_range':[(1,1)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenizer_porter],
              'clf__penalty':['l1','l2'],
              'clf__C':[1.0,10.0,100.0]},
              {'vect__ngram_range':[(1,1)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenizer_porter],
              'vect__use_idf':[False],
              'vect__norm':[None],
              'clf__penalty':['l1','l2'],
              'clf__C':[1.0,10.0,100.0]},
              ]
lr_tfidf= Pipeline([('vect',tfidf),
                    ('clf',LogisticRegression(random_state = 0,
                                              solver ='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,
                           scoring = 'accuracy',
                           cv=5,
                           verbose = 2,
                           n_jobs = 1)
gs_lr_tfidf.fit(X_train,y_train)

gs_lr_tfidf.best_params_
gs_lr_tfidf.best_score_
clf = gs_lr_tfidf.best_estimator_
clf.score(X_test,y_test)


#online algorithms and out-of-core learning
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = (re.sub('[\W]+', ' ' ,text.lower())+' '.join(emoticons).replace('-',''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)#skip header
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text,label
            
next(stream_docs(path='movie_data.csv'))
#get_minibatch to return particular number of documents
def get_minibatch(doc_stream,size):
    docs,y =[] , []
    try:
        for _ in range (size):
            text,label =next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect= HashingVectorizer(decode_error = 'ignore',
                        n_features =2**21,
                        preprocessor = None,
                        tokenizer = tokenizer)
clf= SGDClassifier(loss='log',random_state =1)
doc_stream = stream_docs(path='movie_data.csv')
#out-of-core learning
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train,y_train = get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()


X_test,y_test= get_minibatch(doc_stream,size = 5000)
X_test = vect.transform(X_test)
clf.score(X_test,y_test)
doc_stream


# LDA " Latent Drichlet Allocation
import pandas as pd
df = pd.read_csv('movie_data.csv',encoding = 'utf-8')
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words = 'english',
                        max_df = 0.1,
                        max_features = 5000)
X=count.fit_transform(df['review'].values)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 10,
                                random_state = 123,
                                learning_method ='batch')
X_topics = lda.fit_transform(X)
lda.components_.shape
n_top_words = 5
feature_names =  count.get_feature_names()
for topic_idx,topic in enumerate(lda.components_):
    print("Topic %d:"%(topic_idx+1))
    print(" ".j)