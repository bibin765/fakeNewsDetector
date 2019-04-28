from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
import re 
import nltk
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
import pandas as pd
import csv
from nltk.tokenize import word_tokenize





def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(text))
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)
def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % index]))
    return sentences
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])

def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        
        review = request.form['review']   
        data = cleanup(review)
        print(data)
        model= Doc2Vec.load("doc2vec.model")
        test_data = word_tokenize(data)
        v1 = model.infer_vector(test_data)
        filename='savedsvms.sav'
        loaded_model = joblib.load(filename)
        v1=v1.reshape(1,-1)
        y_pred= loaded_model.predict(v1)
        y_pred =int(y_pred)
        print(y_pred)
        if y_pred == 1 :
            return render_template('real.html')
        else:
            return render_template('fake.html')

if __name__ == "__main__":
    app.run()