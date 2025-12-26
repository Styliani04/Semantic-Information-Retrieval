import pandas as pd
import sys
import re as regex 
import numpy as np  
from sentence_transformers import SentenceTransformer # pip install sentence-transformers, pip install tf-keras
import faiss # pip install faiss-cpu


sys.stdout.reconfigure(encoding='utf-8')

#1. Preprocessing the texts
df = pd.read_csv("Phase2/documents.csv")

df["Text"] = df["Text"].fillna("")
def preprocess_text(text: str):
    text = text.strip()                      
    text = regex.sub(r"\s+", " ", text)        
    text = regex.sub(r"\n+", " ", text)
    return text

df["preprocessed_text"] = df["Text"].apply(preprocess_text)
df = df[df["preprocessed_text"].str.len() > 0]

documents = df["preprocessed_text"].tolist()

#2. Text to dense-vectors/embeddings conversion with DistilBERT
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

#This takes a while to load the model and compute the embeddings, like 20-30 minutes, so we saved them in "document_embeddings.npy" when we ran it the first time.
#If you don't want to wait, just comment out lines 31-35 and uncomment lines 37-38 to load the precomputed embeddings.
embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
)

# np.save("document_embeddings.npy", embeddings)
# embeddings = np.load("document_embeddings.npy")


#3. Building the FAISS index with IndexFlatL2
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#4. Work on queries.csv

queries_df = pd.read_csv("Phase2/queries.csv")
queries = queries_df["Text"].tolist()

for query in queries:
    #a) Calculate the embedding
    embedding = model.encode([query])
    #b) Faiss search with L2 distance
    distances, indices = index.search(embedding, 50)

    print(f"\n{'='*50}")
    print(f"ΤΕΧΤ: {query}")
    print(f"{'='*50}")

    for k in [20, 30, 50]:    
        print(f"\n--- Top {k} Results ---")
        
        #c) Display results in descending order of similarity (ascending order of L2 distance)
        for i in range(k):
            id = indices[0][i]
            distance = distances[0][i]
            similarity = 1 / (1 + distance)
            print(f"Rank {i+1:2}: Doc ID {id} | Similarity: {similarity} | Distance: {distance}")
