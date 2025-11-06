
import os
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

client = Elasticsearch("http://localhost:9200")

# downloading the proper material for the preprocessing of the texts in documents.csv

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download('wordnet')

# the preprocessing function

def preprocess_english_text(text: str):
    # Normalize case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words) 
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)

    preprocessed_words = []

    for word in word_tokens:
        lemma = lemmatizer.lemmatize(word)
        stem = stemmer.stem(lemma)
        preprocessed_words.append(stem)
    
    return " ".join(preprocessed_words)

# we find our documents.csv file in order to extract the texts
file_directory = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(file_directory, "documents.csv")

df = pd.read_csv(file)

# we want to process only the texts, not the ids

texts = df["Text"]

preprocessed_texts = []


for text in texts:
    preprocessed_texts.append(preprocess_english_text(text)
)

# we create a new csv file with the preprocessed texts

preprocessed_text_df = pd.DataFrame({
    "ID" : df["ID"],
    "Text": preprocessed_texts
})

preprocessed_text_df.to_csv("preprocessed_documents.csv",index=False)

# we create a json file to use it in the indexing

