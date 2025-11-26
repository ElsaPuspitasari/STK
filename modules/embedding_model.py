# ============================================================
# 🔥 FINAL — Generate E5 Embedding (CORPUS + QUERY in 1 FILE)
# ============================================================

import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

start_time = time.time()
print("🚀 Memulai proses embedding E5 untuk CORPUS + QUERY...\n")

# ------------------------------------------------------------
# 1) Load Data
# ------------------------------------------------------------
DATA_FILE = "DATASET TANYA JAWAB CLEAN_QA.xlsx"

df = pd.read_excel(DATA_FILE)
questions = df["question"].astype(str).tolist()
answers   = df["answer"].astype(str).tolist()

print(f"📄 Total pasangan Q–A: {len(df)}\n")

# ------------------------------------------------------------
# 2) Load E5 Model
# ------------------------------------------------------------
MODEL_NAME = "intfloat/multilingual-e5-base"
print(f"🔍 Memuat model: {MODEL_NAME}\n")
model = SentenceTransformer(MODEL_NAME)

# ------------------------------------------------------------
# 3) Embedding ANSWER → Corpus utama (passage)
# ------------------------------------------------------------
print("🔧 Menghasilkan embedding ANSWER sebagai PASSAGE...")

answers_prefixed = ["passage: " + a for a in answers]

corpus_embeddings = model.encode(
    answers_prefixed,
    convert_to_numpy=True,
    show_progress_bar=True
)

# NORMALISASI L2
corpus_embeddings = normalize(corpus_embeddings, norm="l2", axis=1)

# ------------------------------------------------------------
# 4) Embedding QUESTION → Query
# ------------------------------------------------------------
print("\n🔧 Menghasilkan embedding QUESTION sebagai QUERY...")

questions_prefixed = ["query: " + q for q in questions]

query_embeddings = model.encode(
    questions_prefixed,
    convert_to_numpy=True,
    show_progress_bar=True
)

query_embeddings = normalize(query_embeddings, norm="l2", axis=1)

# ------------------------------------------------------------
# 5) Simpan semua dalam satu file .npz
# ------------------------------------------------------------
OUT_FILE = "embeddings_all.npz"

np.savez(
    OUT_FILE,
    corpus_embeddings = corpus_embeddings,  
    query_embeddings  = query_embeddings,   
    answers           = np.array(answers, dtype=object),
    questions         = np.array(questions, dtype=object),
)

print(f"\n✅ Semua EMBEDDING disimpan ke: {OUT_FILE}")
print("   - corpus_embeddings (passage = jawaban)")
print("   - query_embeddings  (query = pertanyaan)")
print("   - answers (teks jawaban)")
print("   - questions (teks pertanyaan)")


# ============================================================
# 6) PREVIEW 5 HASIL EMBEDDING (untuk laporan / artikel)
# ============================================================

print("\n\n================= 🟢 SAMPLE 5 QUERY EMBEDDINGS (QUESTION) =================")
for i in range(5):
    print(f"\n➡ QUERY {i+1}")
    print("Teks Pertanyaan :", questions[i][:150], "...")
    print("5 dimensi pertama:", query_embeddings[i][:5])

print("\n\n================= 🔵 SAMPLE 5 PASSAGE EMBEDDINGS (ANSWER) =================")
for i in range(5):
    print(f"\n➡ PASSAGE {i+1}")
    print("Teks Jawaban :", answers[i][:150], "...")
    print("5 dimensi pertama:", corpus_embeddings[i][:5])

# ------------------------------------------------------------
# 7) Summary
# ------------------------------------------------------------
print("\n🎉 Selesai membuat embedding E5 (CORPUS + QUERY)!")
print(f"⏱ Total waktu: {round(time.time() - start_time, 2)} detik")
