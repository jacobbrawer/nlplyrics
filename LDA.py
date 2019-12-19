import pandas as pd
import numpy as np
import requests
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import lyricsgenius as genius
import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from collections import Counter
from os import path
from PIL import Image
import pickle
import json
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


final_songs = pd.read_csv("final_songs.csv", index_col=0)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

doc_set = final_songs['Corpus'].tolist()

stops = ["love", "time", "day", "night", "girl", "baby", "babi", "like", "chorus", "verse", "bridge", "yeah", "whoa", "because", "come", "nigga", "thing"]

texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    raw = re.sub(r'\b\w{1,3}\b', '', i)
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    stopped_tokens = [i for i in tokens if not i in stops]

    
    # # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    stemmed_tokens = [i for i in stemmed_tokens if not i in stops]
    # stemmed_tokens = stopped_tokens
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics= 9, id2word = dictionary, passes=20)

lda_results = [(0, '0.136*"heart" + 0.025*"dream" + 0.021*"fire" + 0.018*"woah" + 0.017*"kiss"'), (1, '0.095*"life" + 0.074*"thing" + 0.049*"world" + 0.027*"song" + 0.019*"people"'), (2, '0.104*"tonight" + 0.042*"dance" + 0.037*"party" + 0.024*"hand" + 0.021*"thing"'), (3, '0.058*"friend" + 0.050*"head" + 0.040*"woman" + 0.034*"lover" + 0.029*"lady", ladies'), (4, '0.095*"shit" + 0.086*"bitch" + 0.016*"bottom" + 0.015*"water" + 0.014*"pussies + 0.014"'), (5, '0.098*"money" + 0.067*"bodies" + 0.022*"type" + 0.016*"bitch" + 0.015*"cash"'), (6, '0.037*"name" + 0.025*"chance" + 0.023*"town" + 0.019*"rain" + 0.017*"tear"'), (7, '0.066*"gang" + 0.056*"taste" + 0.039*"girlfriend" + 0.028*"wrist" + 0.019*"chain"'), (8, '0.109*"thunder" + 0.040*"star" + 0.033*"murder" + 0.022*"ghost" + 0.020*"wish"')]

