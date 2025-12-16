import subprocess
import sys
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd

# initializing our elasticsearch port
client = Elasticsearch("http://localhost:9200")

# 1. Preparing the data

# initializing the encoding to not have the UnicodeEncodeError when printing the texts
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("documents.csv")

# creating a jsonl file from our csv file
with open("documents.jsonl", "w",  encoding="utf-8") as file:
    for _, row in df.iterrows():
        json_data = {"ID": row["ID"], "Text": row["Text"]}
        file.write(json.dumps(json_data) + "\n")

print(f"Created documents.jsonl")

# 2. Creating the index
# Creating our mapping with standard analyzer and BM25 similarity
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

# Creating an index with the mapping above
if client.indices.exists(index="my_texts"):
    client.indices.delete(index="my_texts")
client.indices.create(index="my_texts", body=mapping)

# reading the jsonl file
documents = [] 
with open("documents.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        documents.append({"_index": "my_texts",
            "_id": data["ID"],
            "_source": data})

# adding the texts into the index with bulk since we have a big dataset
bulk(client, documents, refresh=True)

# 3. Function to run queries and save results in TREC format
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
                
# Running queries for k = 20, 30, 50 and saving results
for k in [20, 30, 50]:
    output_file = f"results_{k}.txt"
    run_queries_and_save_results(k, output_file)
    print(f"Created {output_file}")
    
#4. Evaluating results using trec_eval    
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