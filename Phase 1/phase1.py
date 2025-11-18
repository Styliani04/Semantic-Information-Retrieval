
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

# initializing our elasticsearch port
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
file_directory2 = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_directory2, "..", "preprocessed_documents.csv")
file2= os.path.abspath(file_path)

df = pd.read_csv(file2)

with open("preprocessed_documents.jsonl", "w",  encoding="utf-8") as file:
    for _, row in df.iterrows():
        json_data = {"ID": row["ID"], "Text": row["Text"]}
        file.write(json.dumps(json_data) + "\n")

# we take each json line from the jsonl file we created from the preprocessed_document.csv file and put it in a list
json_data = [] 
with open("preprocessed_documents.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        json_data.append(data)

# creating our mapping with standard analyser and BM25 similarity
mapping = {
    "settings":{
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "english"
                },
                "default_search": {
                    "type": "english"
                }
            }
        },
        "index":{
            "similarity":{
                "bm25_similarity":{
                    "type": "BM25"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "ID":{
                    "type": "text",
                    "copy_to": "allContent",                
                    "similarity": "bm25_similarity"
                },
            "Text":{
                    "type": "text",
                    "copy_to": "allContent",                
                    "similarity": "bm25_similarity"
                   },
        }
    }
}

# creating an index with the mapping above
if client.indices.exists(index="my_texts"):
    client.indices.delete(index="my_texts")

client.indices.create(index="my_texts", body=mapping)

# adding the texts into the index with bulk since we have a big dataset
documents = []
for doc in json_data:
    documents.append({
        "_index": "my_texts",
        "_id": doc["ID"],  
        "_source": {
            "Text": doc["Text"] 
        }
    })

bulk(client, documents, refresh=True)

# function for searching in a nice format //might not be needed
def pretty_search_response(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            text = hit["_text"]
            
            pretty_output = f"\nID: {id}\nText: {text}"
            print(pretty_output)

queries = pd.read_csv("queries.csv")

for query in queries["Text"]:
    for k in [20, 30, 50]:
        response = client.search(
            index="my_texts",
            size=k,
            query={
                "match": {
                    "Text": query,
                }
            }
        )
    pretty_search_response(response)    