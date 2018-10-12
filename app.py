#!/Path/to/python

# Flask imports
from flask import Flask, render_template, flash, request, redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea

#Scientific Computing Package Imports
import pandas as pd
import numpy as np
import scipy
import matplotlib as pl
import matplotlib.pyplot as plt

#Scikit-Learn Imports
import pickle
import sklearn as skl
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin ,clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, mean_squared_error, confusion_matrix , auc, accuracy_score

#NLP Imports
import string
import re
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


#Random Number Imports
import random
from time import time
from scipy.stats import randint as sp_randint




## Functions to tokenize and Stem Lyrics
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems



## Class to endow the custom enemble classification model with fit,
## transform, and predict methods.
class AveragingModels(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and
    # employ a max voting for the ensembled prediction
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return scipy.stats.mode(predictions, axis=1)[0]

    def predict_proba(self, X):
        predictions = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_

        ])
        return np.mean(predictions, axis=1)


## Loading picked models for
##       word count CountVectorizer
##       TF-IDF transformer
##       Ensemble Lyrics-Valence Classifier
count_vect = pickle.load(open('count_vect', 'rb'))
lyrics_tfidf = pickle.load(open('lyrics_tfidf', 'rb'))
voting_model = pickle.load(open('voting_model', 'rb'))


##
def classify(user_lyrics):
    tok_lyrics = tokenize(user_lyrics.lower().replace('.',' ').replace(',',' '))
    tok_lyrics = [' '.join(tok_lyrics)]

    lyrics = count_vect.transform(tok_lyrics).toarray()
    lyrics = lyrics_tfidf.transform(lyrics).toarray()
    prediction = voting_model.predict(lyrics)[0][0]
    if prediction == 0:
        return  'low-valence'
    else:
        return 'high-valence'



low_val = pd.read_csv('low_val.csv')
hig_val = pd.read_csv('hig_val.csv')

def get_chord_prog(valence = None):
    low_val_rand = random.sample(range(0, low_val.sort_values(by=['valence']).shape[0]), 5)
    hig_val_rand = random.sample(range(0, hig_val.sort_values(by=['valence']).shape[0]), 5)
    if valence ==  'low-valence':
        return low_val.loc[low_val_rand].drop(labels = ['chord_prog', 'valence'],
        axis = 1).reset_index(drop = True).to_html(justify = 'center',col_space = 50, index = False)
    elif valence ==  'high-valence':
        return hig_val.loc[hig_val_rand].drop(labels = ['chord_prog', 'valence'],
        axis = 1).reset_index(drop = True).to_html(justify = 'center',col_space = 50, index = False)
    elif valence == None:
        return None



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = ''  #Replace with a Secret_Key




class ReusableForm(Form):
    name = TextAreaField('Name:', widget=TextArea(),
        default=u"please type content...")


@app.route('/', methods=['GET', 'POST'])
def main():
    resp = None
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        name = request.form['name']
        resp = classify(name)
        if resp == 'high-valence':
            if form.validate():
            # Save the comment here.
                flash('Based off of your lyric input you should play chords with a high emotional valence. These chord progressions can voice')
                flash('a range of emotions from serenity to happiness. To receive more options for chord progressions just press submit again.')
        elif resp == 'low-valence':
            if form.validate():
            # Save the comment here.
                flash('Based off of your lyric input you should play chords with a low emotional valence. These chord progressions can voice')
                flash('a range of emotions from angry to sad. To receive more options for chord progressions just press submit again.')

            else:
                flash('All the form fields are required. ')
    return render_template('index.html', form=form, dataframe = get_chord_prog(resp))



if __name__ == '__main__':
    app.run(debug=True)
