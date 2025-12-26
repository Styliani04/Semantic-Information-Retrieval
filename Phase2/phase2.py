import pandas as pd
import sys
import re as regex 
import numpy as np  
from sentence_transformers import SentenceTransformer # pip install sentence-transformers, pip install tf-keras
import faiss # pip install faiss-cpu
import subprocess
from elasticsearch import Elasticsearch

client = Elasticsearch("http://localhost:9200")
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

#5. Evaluating results using trec_eval    
def run_queries_and_save_results(k, output_file):
    queries = pd.read_csv("queries.csv")

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries.iterrows():
            query_id = row["ID"]
            query_text = row["Text"]

            response = client.search(
                index="my_texts",
                size=k,
                track_total_hits=False,
                query={
                    "match": {
                        "Text": query_text
                    }
                }
            )
            
            # Sort hits by score in descending order
            hits_sorted = sorted(response["hits"]["hits"], key=lambda x: x["_score"], reverse=True)

            rank = 1
            for hit in hits_sorted:
                doc_id = hit["_id"]
                score = hit["_score"]

                f.write(f"{query_id} Q0 {doc_id} {rank} {score} myIRmethod\n")
                rank += 1
                
for k in [20, 30, 50]:
    output_file = f"results_{k}.txt"
    run_queries_and_save_results(k, output_file)
    print(f"Created {output_file}")

qrels = "qrels.txt"
run_files = {
    "20": "results_20.txt",
    "30": "results_30.txt", 
    "50": "results_50.txt"
}

metrics = ["map", "P.5,10,15,20"] 

for k, run_file in run_files.items():
    output_file = f"eval_{k}.txt"
    
    cmd = ["trec_eval"]
    for m in metrics:
        cmd.extend(["-m", m])
    cmd.extend([qrels, run_file])   # trec_eval -m map -m P.5,10,15,20 qrels.txt results_k.txt > eval_k.txt

        
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) # Running trec_eval command with silenced warnings
        
    if result.returncode == 0: # If trec_eval ran successfully
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        print(f"Created {output_file}")
    else:
        print(f"Error running trec_eval for k={k}")