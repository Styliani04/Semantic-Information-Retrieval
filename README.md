[README.md](https://github.com/user-attachments/files/32539813/README.md)
# Semantic Information Retrieval

A Python information retrieval project comparing **lexical search**, **dense semantic search**, and **hybrid ranking** on a shared document collection and query set. Each method produces ranked results in TREC format and is evaluated using relevance judgments and `trec_eval`.

## Approaches

| Phase | Method | Implementation |
| --- | --- | --- |
| 1 | Lexical retrieval | Elasticsearch with BM25 and an English text analyzer |
| 2 | Dense semantic retrieval | Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS |
| 3 | Hybrid ranking | BM25 candidate retrieval followed by a weighted combination of lexical and semantic scores |

### Phase 1: BM25

Documents are read from CSV, converted to JSONL, and indexed in Elasticsearch under `my_texts`. The `Text` field uses the English analyzer and BM25 similarity. Each query retrieves the top 20, 30, and 50 documents.

### Phase 2: Semantic search

Document text is cleaned by normalizing whitespace and removing empty entries. The `all-MiniLM-L6-v2` model represents documents and queries as 384-dimensional embeddings. Vectors are L2-normalized and searched with FAISS `IndexFlatIP`, making the inner-product scores equivalent to cosine similarity.

The supplied script loads precomputed document embeddings by default. It also contains a commented block for generating them from scratch.

### Phase 3: Hybrid ranking

BM25 retrieves up to 200 candidate documents per query. The candidate documents are encoded with the same Sentence Transformer model and assigned semantic similarity scores.

The final ranking combines min-max-normalized BM25 scores with cosine similarity:

```text
final_score = 0.7 × normalized_BM25 + 0.3 × cosine_similarity
```

The top 20, 30, and 50 candidates are exported for evaluation. This method reranks BM25 candidates rather than searching the entire collection semantically.

## Recorded results

The following values come from the evaluation files included with the project. They have not been regenerated for this README.

### Mean Average Precision by retrieval depth

| Method | Top 20 run | Top 30 run | Top 50 run |
| --- | ---: | ---: | ---: |
| BM25 | 0.6895 | 0.7471 | 0.7701 |
| Semantic search | 0.3129 | 0.3413 | 0.3744 |
| Hybrid ranking | 0.6824 | 0.7493 | **0.7752** |

These are `trec_eval` MAP values computed on runs truncated at the indicated retrieval depth.

### Precision from the top 50 runs

| Method | P@5 | P@10 | P@15 | P@20 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8800 | 0.7300 | 0.7000 | **0.6300** |
| Semantic search | 0.6600 | 0.5000 | 0.3933 | 0.3500 |
| Hybrid ranking | **0.9000** | **0.7700** | 0.7000 | 0.6050 |

On these recorded runs, hybrid ranking slightly improves MAP at depth 50 and precision at ranks 5 and 10 over BM25. BM25 has higher P@20. Dense retrieval alone performs worse on this collection; these results do not establish a general ranking of the methods on other datasets.

## Project files

| Location | Contents |
| --- | --- |
| `Phase1/phase1.py` | Elasticsearch indexing, BM25 retrieval, and evaluation |
| `Phase2/phase2.py` | Embedding-based retrieval with FAISS and evaluation |
| `Phase3/phase3.py` | BM25 candidate retrieval, hybrid ranking, and evaluation |
| Each phase: `documents.csv` | Document collection with `ID` and `Text` columns |
| Each phase: `queries.csv` | Queries with `ID` and `Text` columns |
| Each phase: `qrels.txt` | Relevance judgments used for evaluation |
| `Phase2/document_embeddings.npy` | Precomputed document vectors required by the default Phase 2 code |
| `results*.txt` / `eval*.txt` | Ranked runs and recorded evaluation results |
| Root PDF files | Accompanying phase reports |

The original archive also includes Windows `trec_eval.exe` and `cygwin1.dll` files in each phase directory.

## Requirements

- A Python environment with `pandas`, `numpy`, `elasticsearch`, `sentence-transformers`, and `faiss-cpu`.
- A running Elasticsearch instance for Phases 1 and 3, with a compatible Python client.
- A working `trec_eval` executable available to the scripts.
- Internet access for the first download of the Sentence Transformer model, unless it is already cached.

The project does not include a pinned dependency lockfile. An example dependency installation command is:

```shell
python -m pip install pandas numpy elasticsearch sentence-transformers faiss-cpu
```

Choose an Elasticsearch Python client version compatible with your server. This command alone does not install or start the Elasticsearch server or install `trec_eval`.

## Setup

### 1. Prepare the input files

Place `documents.csv`, `queries.csv`, and `qrels.txt` in each phase directory, as expected by the existing relative paths. If large data files are omitted from your repository copy, obtain the original coursework dataset separately before running the scripts. No dataset download routine is included.

`documents.jsonl` is generated automatically by Phases 1 and 3 and does not need to be provided beforehand.

### 2. Configure Elasticsearch

Phases 1 and 3 currently connect to:

```python
client = Elasticsearch("http://localhost:9200")
```

Adapt this connection to your own server's authentication and TLS settings if required.

**Both scripts delete and recreate the `my_texts` index if it already exists.** Use a dedicated experiment index and run these phases separately.

In the supplied `Phase3/phase3.py`, the `client.search(...)` call omits the index. Before running, add the index argument to restrict candidate retrieval to the project's documents:

```python
response = client.search(
    index="my_texts",
    size=200,
    track_total_hits=False,
    query={"match": {"Text": query_text}}
)
```

### 3. Prepare Phase 2 embeddings

If `Phase2/document_embeddings.npy` is present, the script loads it directly. It must match the current document collection and the row order after preprocessing.

If it is absent, replace the commented encoding block and the `np.load(...)` line with:

```python
embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True
)
np.save("document_embeddings.npy", embeddings)
```

For subsequent runs with unchanged data, you can restore:

```python
embeddings = np.load("document_embeddings.npy")
```

Regenerate embeddings whenever the document contents, preprocessing, ordering, or embedding model changes.

### 4. Make trec_eval available

Each script calls `trec_eval` through `subprocess`. Ensure the executable can be found on `PATH`, or change the command to its explicit path.

For example, when using the bundled Windows executable from its phase directory:

```python
cmd = ["./trec_eval.exe"]
```

Keep its accompanying `cygwin1.dll` available. On Linux or macOS, use a native `trec_eval` installation instead of the Windows executable.

## Run

Run each script with its own phase directory as the working directory. From the repository root, run the following blocks separately:

**Phase 1**

```shell
cd Phase1
python phase1.py
cd ..
```

**Phase 2**

```shell
cd Phase2
python phase2.py
cd ..
```

**Phase 3**

```shell
cd Phase3
python phase3.py
cd ..
```

Phase 3 rebuilds its own Elasticsearch index and computes candidate embeddings, so it does not require running Phase 1 or Phase 2 first.

## Evaluation and output

Each phase writes three ranked runs, with retrieval depths of 20, 30, and 50, followed by evaluation files. Existing files with the same names are overwritten.

TREC run records contain:

```text
query_id Q0 document_id rank score run_name
```

The scripts request MAP and precision at ranks 5, 10, 15, and 20. For example, a manual evaluation command from `Phase1/` is:

```shell
trec_eval -m map -m P.5,10,15,20 qrels.txt results_50.txt
```

## Implementation notes

- Phase 2 depends on a matching embedding file unless its encoding block is enabled.
- Phase 3 assumes BM25 returns at least one candidate; handling queries with no matches would make it more robust.
- The scripts suppress `trec_eval` standard error. If evaluation fails, run the command manually or temporarily stop suppressing stderr to inspect the cause.
- The original scripts use working-directory-relative paths and Windows evaluation binaries. Additional environment setup may be needed on another machine.
- This README documents the supplied source and recorded results. A complete retrieval run and dependency compatibility test were not performed for this documentation update.
