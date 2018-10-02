import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import urllib2
import base64
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')



clean_text = [] #dataframe to store all clean texts 
scores = [] #dataframe to store all corresponding scores 
X_train = []
X_test = []
y_train = []
y_test = []

def cleantext(filename):
    mywords = ['the', 'a', 'is','are','for','that','as','it','to','be']
    from sklearn.feature_extraction.text import TfidfTransformer
    df = pd.read_csv(filename)
    df = df.drop_duplicates(['title'], keep='last')
    df = df.reset_index()
    print df.shape
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', 
                                 stop_words=mywords)
    for i in range(len(df['ARTICLE CONTENT'])-1):
        if str(df['momentum'].loc[i]) != "nan":
            article = str(unicode(str(df['ARTICLE CONTENT'].loc[i]), 
                                  errors='ignore'))
            clean_text.append(article)
            scores.append(int(df['momentum'].loc[i]))
   
    X_train, X_test, y_train, y_test = train_test_split(clean_text, scores)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_train = TfidfTransformer().fit_transform(X_train)
    row, col = X_train.shape #(1900,30495)
    #print X_test.shape #(476, 30495)
    #print len(y_train) #1900
    #print len(y_test) #476
   
    #Linear_SVR(X_train, X_test, y_train,y_test)
    #RFR(X_train, X_test, y_train,y_test, col)
    DecisionTree(X_train, X_test, y_train,y_test)
    LogisticReg(X_train, X_test, y_train,y_test)
    Linear_SVR(X_train, X_test, y_train,y_test)

def DecisionTree(Xtrain,Xtest,ytrain,ytest):
    clf = DecisionTreeRegressor(max_depth=1000, max_features=2)
    clf.fit(Xtrain,ytrain)    
    y_pred = clf.predict(Xtest)
    #print clf.score(y_test, y_pred)
    print"Decision Tree Regressor", metrics.r2_score(ytest, y_pred)   
    scores = cross_val_score(DecisionTreeRegressor(), Xtest, ytest,cv=50)
    print "Cross Val Score:", scores 
    print "MAX SCORE: ", max(scores), "AVERAGE: ", sum(scores)/len(scores)

def LogisticReg(Xtrain,Xtest,ytrain,ytest):
    clf = LogisticRegression()
    clf.fit(Xtrain,ytrain)    
    y_pred = clf.predict(Xtest) 
    print"Logistic Regression", metrics.accuracy_score(ytest, y_pred)   

def Linear_SVR(Xtrain,Xtest,ytrain,ytest):
    cv_scores = []
    #parameters = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    parameters = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
    for i in parameters:
        clf = LinearSVR(loss = 'epsilon_insensitive',C = i)
        clf.fit(Xtrain,ytrain)    
        y_pred = clf.predict(Xtest)
        #print clf.score(y_test, y_pred)   
        cv_scores.append(metrics.r2_score(ytest, y_pred))
    print "CV_SCORES IS: ", cv_scores
    print ("LinearSVR")
    print sum(cv_scores)/float(len(cv_scores)) 

    
def RFR(Xtrain,Xtest,ytrain,ytest, col):
    
    #parameters = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    parameters = [2,5,10,15,20]
    for i in parameters:
        cv_scores = []
        clf = RandomForestRegressor(n_estimators = i, max_features='log2',min_samples_split=2
                                    ,max_depth=None)
        #y_pred = clf.predict(Xtest)
        scores = cross_val_score(clf, Xtest, ytest,cv = 10)   
        #print clf.score(y_test, y_pred)   
        cv_scores.append(scores)
        print scores
        print sum(scores)/float(len(scores)) 
       
    print ("RFR")
    print scores.mean()

def main():
    cleantext("out_normal.csv")

    
    
if __name__ == "__main__":
    main()

