
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

nltk.download("stopwords")
nltk.download("punkt_tab")

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
    return text

file_directory = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(file_directory, "documents.csv")

df = pd.read_csv(file)

texts = df["Text"]

preprocessed_texts = []

for text in texts:
    preprocessed_texts.append(preprocess_english_text(text)
)

preprocessed_text_df = pd.DataFrame({
    "ID" : df["ID"],
    "Text": preprocessed_texts
})

preprocessed_text_df.to_csv("preprocessed_documents.csv",index=False)
