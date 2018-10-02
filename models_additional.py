import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
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
    mywords = ['the', 'a', 'is','are','for','that','as','it','to','be', 
               '<h>', '<p>']
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
    row, col = X_train.shape 
  
   
    #Linear_SVR(X_train, X_test, y_train,y_test)
    #RFR(X_train, X_test, y_train,y_test, col)
    #DecisionTree(X_train, X_test, y_train,y_test)
    LassoReg(X_train, X_test, y_train,y_test)

def Linear_SVR(Xtrain,Xtest,ytrain,ytest):
    cv_scores = []
    parameters = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
    for i in parameters:
        clf = LinearSVR(loss = 'squared_epsilon_insensitive',C = i)
        clf.fit(Xtrain,ytrain)    
        y_pred = clf.predict(Xtest)
        #print clf.score(y_test, y_pred)   
        cv_scores.append(metrics.r2_score(ytest, y_pred))
    print ("LinearSVR")
    print sum(cv_scores)/float(len(cv_scores)) 
    
    
def RFR(Xtrain,Xtest,ytrain,ytest, col):
    parameters = [50,100,150,200,250]
    for i in parameters:
        cv_scores = []
        clf = RandomForestRegressor(n_estimators = i, max_features='log2',min_samples_split=2,max_depth=20).fit(Xtrain,ytrain)
        #y_pred = clf.predict(Xtest)
        scores = cross_val_score(clf, Xtest, ytest,cv = 10)   
        cv_scores.append(scores)
        print scores
        print sum(scores)/float(len(scores)) 
    print ("RFR")
    print scores.mean()


def DecisionTree(Xtrain,Xtest,ytrain,ytest):
    cv_scores = []
    parameters = [10,20,50,100,200]
    for i in parameters:
        clf = DecisionTreeRegressor(max_depth = i)
        clf.fit(Xtrain,ytrain)    
        y_pred = clf.predict(Xtest)
        #print clf.score(y_test, y_pred)   
        cv_scores.append(metrics.r2_score(ytest, y_pred))
        print sum(cv_scores)/float(len(cv_scores))
    print ("DecisionTreeRegressor")


def LassoReg(Xtrain,Xtest,ytrain,ytest):
    cv_scores = []
    alphaParams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    for i in alphaParams:
        clf = Lasso(alpha = i)
        clf.fit(Xtrain,ytrain)
        scores = cross_val_score(clf, Xtest, ytest,cv = 10)   
        cv_scores.append(scores.mean())
        #print i, metrics.r2_score(ytest, y_pred)
    print ("Lasso Regression")
    print sum(cv_scores)/float(len(cv_scores)) 
    print "DONE\n"    
        


def main():
    cleantext("out_opoint.csv")

    
    
if __name__ == "__main__":
    main()

