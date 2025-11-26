# =============================================================
# PRECISION@K versi proporsional + TAMPILKAN TEKS QUERY
# =============================================================

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
EMB_FILE   = "embeddings_all.npz"

print("Loading model & embeddings...")
model = SentenceTransformer(MODEL_NAME)
data = np.load(EMB_FILE, allow_pickle=True)

corpus_emb = data["corpus_embeddings"]
questions  = data["questions"]

# K seperti tabel skripsi
K_LIST = [5, 10, 15, 20, 25]

# Threshold relevansi
THRESHOLD = 0.85

# Sampel yang ditampilkan
N_SAMPLE = 5

def retrieve(question):
    q = "query: " + question
    q_emb = model.encode(q, convert_to_numpy=True, normalize_embeddings=True)

    scores = np.dot(corpus_emb, q_emb)
    ranked = scores.argsort()[::-1]
    return ranked, scores


print("\n==================== Precision@K ====================\n")

# Header tabel: tambah kolom Query Text
header = f"{'No':<3} | {'Query Text':<40} | " + " | ".join([f"P@{k}" for k in K_LIST])
print(header)
print("-" * len(header))

table = []

for i in range(N_SAMPLE):
    q_text = questions[i][:40].replace("\n", " ")  # potong 40 char untuk rapi

    ranked, scores = retrieve(questions[i])
    row_scores = []

    for K in K_LIST:
        top_scores = scores[ranked[:K]]
        num_rel = np.sum(top_scores >= THRESHOLD)
        row_scores.append(round(num_rel / K, 2))

    table.append(row_scores)

    # Print tabel dengan teks query
    row_str = f"{i+1:<3} | {q_text:<40} | " + " | ".join([f"{p:<4}" for p in row_scores])
    print(row_str)

# Rata-rata
avg = np.mean(table, axis=0)

print("\nRata-rata:")
for k, v in zip(K_LIST, avg):
    print(f"P@{k}: {round(v, 2)}")
