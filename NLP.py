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
from keras.models import model_from_json
import pickle
import json

def collect_songs_from_billboard(start_year,end_year):
    '''This function takes in a start year and and end year, then iterates through each year to 
    pull song data from billboard or bobborst as needed. Then it uses beautiful soup to clean
    the data. Finally it stores the cleaned data in a dataframe and returns it
    
    Parameters:
    
    start_year (int): the year to start at.
    end_year (int): the year to end at.
    Returns: 
    
    dataframe.
    '''
    years = np.arange(start_year, end_year + 1).astype(int)
    dataset = pd.DataFrame()
    url_list = []
    all_years = pd.DataFrame()
    final_years = np.arange(2013,2020)
    ### Billboard doesn't have it's own complete results from 1970 to 2016,
    ### so we'll use bobborst.com as our primary and collect from billboard as needed
    #URL Constructor
    for i in range (0, len(years)):
        url_list.append("http://billboardtop100of.com/" + str(years[i]) + "-2/")      
    for i in range(0, len(url_list)):
        if years[i] in final_years:
            sys.stdout.write("\r" + "Collecting Songs from " +str(years[i]) + " via https://www.billboard.com")
            sys.stdout.flush()
            url = "https://www.billboard.com/charts/year-end/" + str(years[i]) + "/hot-100-songs"
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            all_ranks = soup.find_all("div", class_="ye-chart-item__rank")
            all_titles = soup.find_all('div', class_="ye-chart-item__title")
            all_artists = soup.find_all("div", class_="ye-chart-item__artist")
            for j in range (0, len(all_ranks)):
                row = {
                    "Rank": all_ranks[j].get_text(strip=True),
                    "Song Title": all_titles[j].get_text(strip=True),
                    "Artist": all_artists[j].get_text(strip=True),
                    "Year": years[i]
                }
                dataset = dataset.append(row, ignore_index=True)
        else:
            sys.stdout.write("\r" + "Collecting Songs from " +str(years[i]) + " via https://www.billboard.com")
            sys.stdout.flush()
            url = "http://billboardtop100of.com/" + str(years[i]) + "-2/"
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            table = soup.find_all('tr')
            for j in range(0, len(table)):
                columns = table[j].find_all('td')
                row = {
                    "Rank": columns[0].get_text(strip=True),
                    "Artist": columns[1].get_text(strip=True),
                    "Song Title": columns[2].get_text(strip=True),
                    "Year": years[i]
                }
                dataset = dataset.append(row, ignore_index=True)
    dataset['Year'] = dataset['Year'].astype(int)
    return dataset


def add_spacy_data(dataset, feature_column):
    '''
    Grabs the verb, adverb, noun, and stop word Parts of Speech (POS) 
    tokens and pushes them into a new dataset. returns an 
    enriched dataset.
    
    Parameters:
    
    dataset (dataframe): the dataframe to parse
    feature_column (string): the column to parse in the dataset.
    
    Returns: 
    dataframe
    '''
    
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_sm')
    ##
    for i in range (0, len(dataset)):
        print("Extracting verbs and topics from record {} of {}".format(i+1, len(dataset)), end = "\r")
        song = dataset.iloc[i][feature_column]
        doc = nlp(song)
        spacy_dataframe = pd.DataFrame()
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }
            spacy_dataframe = spacy_dataframe.append(row, ignore_index = True)
        verbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
        nouns.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
        corpus_clean = " ".join(spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
    dataset['Verbs'] = verbs
    dataset['Nouns'] = nouns
    dataset['Adverbs'] = adverbs
    dataset['Corpus'] = corpus
    return dataset

lyric_output = []

def pre_clean(dataset):
    for i in range(0,len(dataset)):
        oldlyric = dataset.iloc[i]['lyrics']
        newlyric = re.sub(r'[^A-Za-z0-9]+', ' ', oldlyric)
        lyric_output.append(newlyric)
    dataset['lyrics'] = lyric_output
    return dataset

def prep_corpus(raw_string):
    '''Single use of add_spacy_data to enable pipelining 
    data into predictions
    
    Parameters:
    raw_string (string): String to be parsed
    
    Returns:
    parsed string
    '''

    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(raw_string)
    spacy_dataframe = pd.DataFrame()
    for token in doc:
        if token.lemma_ == "-PRON-":
                lemma = token.text
        else:
            lemma = token.lemma_
        row = {
            "Word": token.text,
            "Lemma": lemma,
            "PoS": token.pos_,
            "Stop Word": token.is_stop
        }
        spacy_dataframe = spacy_dataframe.append(row, ignore_index = True)
    verbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
    nouns.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
    adverbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
    corpus_clean = " ".join(spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
    corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   

    return corpus_clean

all_songs = collect_songs_from_billboard(1970, 2019)

all_songs["Artist"][all_songs['Artist'] == "Jackson 5"] = "The Jackson 5"
all_songs["Artist"][all_songs['Artist'] == "Beatles"] = "The Beatles"

api = genius.Genius("Gk7JH9g31J9T2nWV-o82WaGwIZQ_04LgbxcJypt4dBRdaGSH494rBORd2qMIVlzJ",sleep_time=0.01, verbose=False)

all_song_data = pd.DataFrame()
start_time = datetime.now()
print("Started at {}".format(start_time))
for i in range(0, len(all_songs)):
    rolling_pct = int((i/len(all_songs))*100)
    print(str(rolling_pct) + "% complete." + " Collecting Record " + str(i) +" of " +
          str(len(all_songs)) +". Year " + str(all_songs.iloc[i]['Year']) + "." + " Currently collecting " + 
          all_songs.iloc[i]['Song Title'] + " by " + all_songs.iloc[i]['Artist'] + " "*50, end="\r")
    song_title = all_songs.iloc[i]['Song Title']
    song_title = re.sub(" and ", " & ", song_title)
    song_title_test = re.sub(r'\W+', '', song_title).lower()
    artist_name = all_songs.iloc[i]['Artist']
    artist_name = re.sub(" and ", " & ", artist_name)

    try:
        song = api.search_song(song_title, artist=artist_name)
        result_title = re.sub(r'\W+', '', song.title).lower()
        if result_title == song_title_test:
            song_album = song.album
            song_album_url = song.album_url
            featured_artists = song.featured_artists
            song_lyrics = re.sub("\n", " ", song.lyrics) #Remove newline breaks, we won't need them.
            song_media = song.media
            song_url = song.url
            song_writer_artists = song.writer_artists
            song_year = song.year
        else:
            print(song_title)
            print(result_title)
            song_album = "null"
            song_album_url = "null"
            featured_artists = "null"
            song_lyrics = "null"
            song_media = "null"
            song_url = "null"
            song_writer_artists = "null"
            song_year = "null"
    except:
        song_album = "null"
        song_album_url = "null"
        featured_artists = "null"
        song_lyrics = "null"
        song_media = "null"
        song_url = "null"
        song_writer_artists = "null"
        song_year = "null"
        
    row = {
        "Year": all_songs.iloc[i]['Year'],
        "Rank": all_songs.iloc[i]['Rank'],
        "Song Title": all_songs.iloc[i]['Song Title'],
        "Artist": all_songs.iloc[i]['Artist'],
        "Album": song_album,
        "Album URL": song_album_url,
        "Featured Artists": featured_artists,
        "Lyrics": song_lyrics,
        "Media": song_media,
        "Song URL": song_url,
        "Writers": song_writer_artists,
        "Release Date": song_year
    }
    all_song_data = all_song_data.append(row, ignore_index=True)
end_time = datetime.now()
print("\nCompleted at {}".format(start_time))
print("Total time to collect: {}".format(end_time - start_time))


all_song_data.to_csv("all_songs_data.csv")
all_song_data.to_json("all_song_data.json", orient='records')
loaded_song_dataset = pd.read_csv("all_songs_data.csv",index_col=0)

songs_with_lyrics_dataset = loaded_song_dataset.dropna(subset=['Lyrics'])
prepared_songs_dataset = add_spacy_data(songs_with_lyrics_dataset, 'Lyrics')

prepared_songs_dataset = prepared_songs_dataset.drop(columns = ['Unnamed: 0'])
word_counts = []
unique_word_counts = []
for i in range (0, len(prepared_songs_dataset)):
    word_counts.append(len(prepared_songs_dataset.iloc[i]['Lyrics'].split()))
    unique_word_counts.append(len(set(prepared_songs_dataset.iloc[i]['Lyrics'].split())))
prepared_songs_dataset['Word Counts'] = word_counts
prepared_songs_dataset['Unique Word Counts'] = unique_word_counts

prepared_songs_dataset = pd.read_csv('prepped_data.csv', index_col=0)

summary_dataset = pd.DataFrame()
years = prepared_songs_dataset['Year'].unique().tolist()
for i in range(0, len(years)):
    row = {
        "Year": years[i],
        "Average Words": prepared_songs_dataset['Word Counts'][prepared_songs_dataset['Year'] == years[i]].mean(),
        "Unique Words": prepared_songs_dataset['Unique Word Counts'][prepared_songs_dataset['Year'] == years[i]].mean()
    }
    summary_dataset = summary_dataset.append(row, ignore_index=True)
summary_dataset["Year"] = summary_dataset['Year'].astype(int)

characteristics = prepared_songs_dataset.groupby('Year').count()




