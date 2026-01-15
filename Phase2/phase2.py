import os
import pandas as pd
import sys
import re as regex 
import numpy as np  
from sentence_transformers import SentenceTransformer # pip install sentence-transformers, pip install tf-keras
import faiss 
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

#1. Preprocessing the texts
df = pd.read_csv("documents.csv")
df["Text"] = df["Text"].fillna("")

def preprocess_text(text: str):
    text = text.strip()                      
    text = regex.sub(r"\s+", " ", text)        
    text = regex.sub(r"\n+", " ", text)
    return text

df["preprocessed_text"] = df["Text"].apply(preprocess_text)
df = df[df["preprocessed_text"].str.len() > 0]

documents = df["preprocessed_text"].tolist()

#2. Text to dense-vectors/embeddings conversion with SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # Dimension: 384

# Uncomment the following lines and comment line 39 to compute embeddings from scratch
'''
embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True
)

np.save("document_embeddings.npy", embeddings)
'''
# Loading pre-computed embeddings
embeddings = np.load("document_embeddings.npy")


#3. Building the FAISS index with IndexFlatL2
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
index.add(embeddings)

#4. Work on queries.csv
queries_df = pd.read_csv("queries.csv")
queries = queries_df["Text"].tolist()
query_ids = queries_df["ID"].tolist()

#a) Encode all queries at once to speed up the process
query_embeddings = model.encode(queries, show_progress_bar=True, batch_size=64)
faiss.normalize_L2(query_embeddings)

for k in [20, 30, 50]:
    
    #b) One-time search for all queries
    distances, indices = index.search(query_embeddings, k)
    
    output_file = f"results2_{k}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, qid in enumerate(query_ids):
            #c) Save results in TREC format
            for rank, (doc_idx, score) in enumerate(zip(indices[i], distances[i]), 1):
                doc_id = df.iloc[doc_idx]["ID"]
                f.write(f"{qid} Q0 {doc_id} {rank} {score} FAISS_TRANSFORMER\n")

#5. Evaluating results using trec_eval       
qrels = "qrels.txt" 

run_files = {
    "20": "results2_20.txt",
    "30": "results2_30.txt", 
    "50": "results2_50.txt"
}

metrics = ["map", "P.5,10,15,20"] 

for k, run_file in run_files.items():
    output_file = f"eval2_{k}.txt"
    
    cmd = ["trec_eval"]
    for m in metrics:
        cmd.extend(["-m", m])
    cmd.extend([qrels, run_file])   # trec_eval -m map -m P.5,10,15,20 qrels.txt results2_k.txt > eval2_k.txt
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) # Running trec_eval command with silenced warnings
    
        if result.returncode == 0: # If trec_eval ran successfully
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
                print(f"Created {output_file}")
        else:
            print(f"Error running trec_eval for k={k}")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred while running trec_eval for k={k}: {e}")
     