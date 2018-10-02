import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
import nltk

import string 
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import urllib2
import base64
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')


clean_text = [] #dataframe to store all clean texts 
scores = [] #dataframe to store all corresponding scores 


def cleantext(filename):
    df = pd.read_csv(filename)
    '''Trying to fix the 9 texts that are not convertible to str type'''
    #for i in range(len(df['ARTICLE CONTENT'])-1):
        #for i in range(len(df['ARTICLE CONTENT'].loc[i].split())):
            #df['ARTICLE CONTENT'].loc[i].split()[i] = str(df['ARTICLE CONTENT'].loc[i].split()[i])
            #if type(df['ARTICLE CONTENT'].loc[i].split()[i]) != str:
                #print df['ARTICLE CONTENT'].loc[i].split()[i], "IS NOT A STRING", type(df['ARTICLE CONTENT'].loc[i].split()[i]), i

    #vectorizer = HashingVectorizer(n_features=20)
    for i in range(len(df['ARTICLE CONTENT'])-1):
        new_tokens = []
        try:
            df['ARTICLE CONTENT'].loc[i] = unicode(df['ARTICLE CONTENT'].loc[i], errors='ignore')
        except:
            print i
            print "HEREEEEE"
            print df['ARTICLE CONTENT'].loc[i]
        tokens = word_tokenize(str(df['ARTICLE CONTENT'].loc[i]).encode('utf-8'))
        #convert to lower case
        tokens = [str(w.lower()) for w in tokens]
        #delete punctuations
        words = [str(word) for word in tokens if word.isalpha()]
        #delete stopwords such as: the, a, is 
        stop_words = set(stopwords.words('english'))   
        words = [str(w) for w in words if not w in stop_words]  
        for j in words:
            if j!= ' ':
                new_tokens.append(j)
        clean_text.append(new_tokens)
        scores.append(float(df['momentum'].values[i]))  
    print "SCORES:", scores
    #print "CLEAN TEXT:", clean_text
    try:
        print "LENGTH 1:", [len(clean_text)]
    except:
        print "DID NOT WORK"
    try:
        print "LENGTH 2:", [len(clean_text[0]), len(clean_text[1]),len(clean_text[2]),len(clean_text[3])]
    except:
        print "DID NOT WORK"        
    SGD_text(clean_text,scores)
    
    
def SGD_text(X,y):
    tempArray=[]
    for i in X:
        for j in i:
            tempArray.append(j)
    X_train, X_test, y_train, y_test = train_test_split(tempArray,y,test_size = 0.2)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    X_train = tfidf.fit_transform(X_train).toarray()
    clf = SGDClassifier(penalty='l2',alpha=1e-3, random_state=42)
    clf.fit(X_train,y_train)     
    y_pred = clf.predict(X_test)
    print "ACCURACY:", clf.score(y_test, y_pred)    
        

def main():
    cleantext("out_normal_moreover.csv")
    SGD_text(X,y)
    
    
if __name__ == "__main__":
    main()

