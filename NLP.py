# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 23:03:48 2018

@author: Lakshay Wadhwa
"""
#importing regex package
import regex as re
#importing numpy array
import numpy as np
#importing pandas array
import pandas as pd
#importing pickle array
import pickle
import heapq
#importing nltk array
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
df1 = pd.read_csv("~/Downloads/training.csv", header=None,sep=',', encoding='latin-1', error_bad_lines=False)
#nltk.download("stopwords")
x=df1[0].values.tolist()
y=df1[1].values.tolist()
x=np.asarray(x)
#negative tweets
#x=df1[:1299]
#positive tweets
#y=df1[813004:814303]
#positive tweets list


#creating the data
from nltk.stem import WordNetLemmatizer
 
word_tokens=[]
corpus=[]
stop_words = set(stopwords.words('english'))
#sentences1=nltk.sent_tokenize(paragraph1)

########################lemmatizing the list
lemmatizer=WordNetLemmatizer()
for i in range(len(y)):
    words1=nltk.word_tokenize(y[i])
    newwords1=[]
    for j in words1:
        j=lemmatizer.lemmatize(j)
        newwords1.append(j)
    y[i]=' '.join(newwords1)    

##########Cleaning the data    
corpus=[]
for i in range(0,len(y)):
    review=re.sub(r'\W',' ',str(y[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)
    #y[i]=' '.join(newwords)
   # y[i]=y[i].lower()
    #y[i]=re.sub(r'\W',' ',y[i])
    #y[i]=re.sub(r'\s',' ',y[i])
    #corpus.append(y[i])
#removing stop words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=19997,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
Y=vectorizer.fit_transform(corpus).toarray()
#####removing the stopwords
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
Y=transformer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(Y,x,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=19997,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
Y=vectorizer.fit_transform(corpus).toarray()
    
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(text_train,sent_train)
sent_pred=classifier.predict(text_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)    
sample=["a Gas Killing Animal who kills his people and enjoys it Animal Assad Animal Assad"]
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample))


sample1=["Nick is a great professor"]
sample1=tfidf.transform(sample1).toarray()
print(clf.predict(sample1))


sample2=["UC boulder is a very competitive school with great professors"]
sample2=tfidf.transform(sample2).toarray()
print(clf.predict(sample2))
