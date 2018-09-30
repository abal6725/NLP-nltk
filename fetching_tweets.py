# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:50:43 2018

@author: Lakshay Wadhwa
"""
import numpy as np
import tweepy
import re
import nltk
from tweepy import OAuthHandler
consumer_key='COsBUcunL1OXee7b7GqOdENEJ'
consumer_secret='yszmA7FHgNZkzbXhdwiLQ0Zs0Pq4QPXCXRjJAViLsspG5pkn4s'
access_token='718866271815643136-iBaqklxcFNWM7mr1D6WO8leAtDM7F0s'
access_secret='HLaqGYWcKU1YWFHSvBCyaxTGktkmNf2lLzuMyGJaKkSaW'
auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
#args=['@realDonaldTrump']
api=tweepy.API(auth,timeout=10)
list_tweets=[]
#query=args[0]

for status in tweepy.Cursor(api.user_timeline,id="@realDonaldTrump",lang='en',result_type='recent').items(100):
    list_tweets.append(status.text)
    
        
print(list_tweets)
#thefile = open('C:/Users/Lakshay Wadhwa/Desktop/LAS/LAS_twitter.txt', 'w')
for i in range(1,len(list_tweets)):
    print(list_tweets[i],end="/n")


from nltk.stem import WordNetLemmatizer
 

#sentences1=nltk.sent_tokenize(paragraph1)
lemmatizer=WordNetLemmatizer()
for i in range(len(list_tweets)):
    words1=nltk.word_tokenize(list_tweets[i])
    newwords1=[]
    for j in words1:
        j=lemmatizer.lemmatize(j)
        newwords1.append(j)
    list_tweets[i]=' '.join(newwords1)      
#removing stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

word_tokens=[]
stop_words = set(stopwords.words('english'))

for i in range(len(list_tweets)):
    words=nltk.word_tokenize(list_tweets[i])
    newwords=[word for word in words if word not in stop_words]
    list_tweets[i]=' '.join(newwords)
   
            
            
    

#filtered_sentence = [w for w in word_tokens if not w in stop_words]


#filtered_sentence = [w for w in word_tokens if not w in stop_words]
    
# creating bag of words

for i in range(len(list_tweets)):
    list_tweets[i]=list_tweets[i].lower()
    list_tweets[i]=re.sub(r'\W',' ',list_tweets[i])
    list_tweets[i]=re.sub(r'\s',' ',list_tweets[i])


    
