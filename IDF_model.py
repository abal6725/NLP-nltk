# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:15:16 2018

@author: Lakshay Wadhwa
"""
##########importing the packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tweepy
import re
import nltk
import heapq
###############calling tweets
from tweepy import OAuthHandler
consumer_key='COsBUcunL1OXee7b7GqOdENEJ'
consumer_secret='yszmA7FHgNZkzbXhdwiLQ0Zs0Pq4QPXCXRjJAViLsspG5pkn4s'
access_token='718866271815643136-iBaqklxcFNWM7mr1D6WO8leAtDM7F0s'
access_secret='HLaqGYWcKU1YWFHSvBCyaxTGktkmNf2lLzuMyGJaKkSaW'
auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
#args=['@realDonaldTrump']
#####calling tweets through tweepy
api=tweepy.API(auth,timeout=10)
list_tweets=[]
predicted_sentiments=[]
##############total_positive_tweets now
total_pos=0
#################total_negative_tweets now
total_neg=0
#query=args[0]
########opening vectorizer
with open('tfidfmodel.pickle','rb') as f:
    vectorizer=pickle.load(f)
#########33opening classifier    
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)   
###############3calling the tweets of donald trump     
for status in tweepy.Cursor(api.user_timeline,id="@tesla",lang='en',result_type='recent').items(5000):
    list_tweets.append(status.text)
    
###########cleaning tweets    
for i in range(len(list_tweets)):
    list_tweets[i]=list_tweets[i].lower()
    list_tweets[i]=re.sub(r'\W',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+https://t.co/[a-zA-Z0-9]*\s',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'^https://t.co/[a-zA-Z0-9]*\s',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+https://t.co/[a-zA-Z0-9]*$',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s+',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\W',' ',list_tweets[i])
    list_tweets[i]=re.sub(r"that's","that is",list_tweets[i])
    list_tweets[i]=re.sub(r"there's","there is",list_tweets[i])
    list_tweets[i]=re.sub(r"what's","what is",list_tweets[i])
    list_tweets[i]=re.sub(r"where's","where is",list_tweets[i])
    list_tweets[i]=re.sub(r"it's","it is",list_tweets[i])
    list_tweets[i]=re.sub(r"who's","who is",list_tweets[i])
    list_tweets[i]=re.sub(r"i's","i am",list_tweets[i])
    list_tweets[i]=re.sub(r"she's","she is",list_tweets[i])
    list_tweets[i]=re.sub(r"he's","he is",list_tweets[i])
    list_tweets[i]=re.sub(r"they're","there are",list_tweets[i])
    list_tweets[i]=re.sub(r"who're","who are",list_tweets[i])
    list_tweets[i]=re.sub(r"ain't","am not",list_tweets[i])
    list_tweets[i]=re.sub(r"wouldn't'","would not",list_tweets[i])
    list_tweets[i]=re.sub(r"shouldn't","should not",list_tweets[i])
    list_tweets[i]=re.sub(r"can't","can not",list_tweets[i])
    list_tweets[i]=re.sub(r"^rt"," ",list_tweets[i])
    predicted_sentiments.append(clf.predict(vectorizer.transform([list_tweets[i]]).toarray()))
    #################predicting the sentiments
    sent=clf.predict(vectorizer.transform([list_tweets[i]]).toarray())
    if sent[0]==4:
        total_pos+=1
    elif sent[0]==0:
        total_neg+=1
    
    
    
    

########3plotting the results

##########taking objects as positive and negative
objects=['Positive','Negative']
y_pos=np.arange(len(objects))
plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel("no. of tweets")
plt.xlabel("negative and positive tweets")
plt.show()    
      
            
    
