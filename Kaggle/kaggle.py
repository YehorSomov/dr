import numpy as np
import pandas as pd
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(nltk.corpus.stopwords.words('english'))

def clear_data(s):
    s = s.strip()
    z = re.findall(r'[A-Za-z]+', s)
    z = [a for a in z if not a in stop_words]
    s = ' '.join(z)
    return s

def main():
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test.fillna(value="unknown", inplace=True)
    test.drop(['id'], axis=1, inplace=True)
    
    
    y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    x_train = train[['comment_text']]
    
    x_train.comment_text = x_train.comment_text.apply(lambda x: clear_data(x))
    test.comment_text = test.comment_text.apply(lambda z: clear_data(z))
    
    vect = TfidfVectorizer()
    vect = vect.fit(x_train['comment_text'])

    x_train_vect = vect.transform(x_train['comment_text'])
    x_test_vect = vect.transform(test['comment_text'])
    
    colums = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y_pred = pd.read_csv('sample_submission.csv')
    
    for i in colums:
        clf = LogisticRegression(C=4, solver='sag')
        clf.fit(x_train_vect, y_train[i])
        y_pred[i] = clf.predict_proba(x_test_vect)[:,1]
        pred_train = clf.predict_proba(x_train_vect)[:,1]
        print('log loss', i, ':', log_loss(y_train[i], pred_train))
    
    y_pred.to_csv("test_submission.csv", index=False)
    
main()

