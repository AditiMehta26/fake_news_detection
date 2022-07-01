import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import flask
import joblib
import pickle
import newspaper
from newspaper import Article
import urllib 
import nltk
nltk.download('punkt')

#Loading flask  and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pk1', 'rb') as handle:
    model = pickle.load(handle)
    
@app.route('/')
def main():
    return render_template('index.html')

#Receiving the input url from the user and usingWeb Scrapping to extract the news content
@app.route('/predict',methods= ['POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    try:
      article = Article(str(url))
      article.download()
      article.parse()
      article.nlp()
      
    except:
        pass
    news = article.summary 
    #Passing the news article to the modeland return in whether it is fake or real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))
 

if __name__=="__main__":
    
    app.run(debug=True)
