import subprocess
import sys
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd 
from sentence_transformers import SentenceTransformer # pip install sentence-transformers, pip install tf-keras
import faiss 

# initializing our elasticsearch port
client = Elasticsearch("http://localhost:9200")

# Copying the code from phase1.py to create the index and add the documents
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("documents.csv")

# creating a jsonl file from our csv file
with open("documents.jsonl", "w",  encoding="utf-8") as file:
    for _, row in df.iterrows():
        json_data = {"ID": row["ID"], "Text": row["Text"]}
        file.write(json.dumps(json_data) + "\n")

print(f"Created documents.jsonl")

mapping = {
    "settings":{
        "analysis": {
            "analyzer": {
                "my_english": {
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
                    "type": "keyword",
                },
            "Text":{
                    "type": "text",
                    "analyzer": "my_english",
                    "similarity": "bm25_similarity"
                   },
        }
    }
}

if client.indices.exists(index="my_texts"):
    client.indices.delete(index="my_texts")
client.indices.create(index="my_texts", body=mapping)

documents = [] 
with open("documents.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        documents.append({"_index": "my_texts",
            "_id": data["ID"],
            "_source": data})

bulk(client, documents, refresh=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

queries = pd.read_csv("queries.csv")

# We will store the results for all k values in a dictionary to write them to files later
all_results = {20: [], 30: [], 50: []}

for _, row in queries.iterrows():
    query_id = row["ID"]
    query_text = row["Text"]

    # Running a BM25 search to get top 200 candidate documents for each query
    response = client.search(  
        size=200,
        track_total_hits=False,
        query={
            "match": {
                "Text": query_text
            }
        }
    )
    hits = response["hits"]["hits"]
    
    bm25_scores = [hit["_score"] for hit in hits]
    
    # Normalizing BM25 scores to [0, 1] range for better combination with cosine similarity scores
    if len(bm25_scores) > 1:
        max_bm = max(bm25_scores)
        min_bm = min(bm25_scores)
        
        # normalize: (score - min) / (max - min)
        norm_bm25 = [(s - min_bm) / (max_bm - min_bm + 1e-9) for s in bm25_scores]
    else:
        norm_bm25 = [1.0] * len(bm25_scores)

    candidate_texts = [hit["_source"]["Text"] for hit in hits]
    candidate_ids = [hit["_id"] for hit in hits]

    # Copying the code from phase2.py to compute semantic similarity scores for the candidate documents and combine them with BM25 scores    
    embeddings = model.encode(
        candidate_texts,
        batch_size=64,
        show_progress_bar=True
    )
    faiss.normalize_L2(embeddings)
    
    query_embeddings = model.encode([query_text], show_progress_bar=True, batch_size=64)
    faiss.normalize_L2(query_embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # We will search for all candidate documents to get their cosine similarity scores with the query, 
    # and then combine these scores with the normalized BM25 scores using a weighted sum to get a final 
    # score for each candidate document. Finally, we will sort the candidates based on this final score.
    num_candidates = len(candidate_texts)
    semantic_distances, indices = index.search(query_embeddings, num_candidates)
    
    semantic_scores_map = {}
    for local_idx, dist in zip(indices[0], semantic_distances[0]):
        semantic_scores_map[local_idx] = dist

    # We will use a weight w to balance the contribution of BM25 and semantic similarity scores.
    w = 0.7  #BM25 had a better performance than semantic similarity in phase1 and phase2, so we give it a higher weight.
    weighted_results = []
    
    for i in range(num_candidates):
        doc_id = candidate_ids[i]
        s_bm25 = norm_bm25[i]
        s_semantic = semantic_scores_map.get(i, 0.0)
        
        final_score = (w * s_bm25) + ((1 - w) * s_semantic)
        weighted_results.append((doc_id, final_score))

    weighted_results.sort(key=lambda x: x[1], reverse=True)
    
    for k in [20, 30, 50]:
        for rank, (doc_id, f_score) in enumerate(weighted_results[:k], 1):
            all_results[k].append(f"{query_id} Q0 {doc_id} {rank} {f_score:.4f} Weighted_Hybrid \n")
            
            
for k in [20, 30, 50]:
    output_file = f"results3_{k}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_results[k]:
            f.write(line)

#2. Evaluating results using trec_eval       
qrels = "qrels.txt" 

run_files = {
    "20": "results3_20.txt",
    "30": "results3_30.txt", 
    "50": "results3_50.txt"
}

metrics = ["map", "P.5,10,15,20"] 

for k, run_file in run_files.items():
    output_file = f"eval3_{k}.txt"
    
    cmd = ["trec_eval"]
    for m in metrics:
        cmd.extend(["-m", m])
    cmd.extend([qrels, run_file])   # trec_eval -m map -m P.5,10,15,20 qrels.txt results3_k.txt > eval3_k.txt
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
     