# ============================================================
# 🔥 TEST MANUAL — Cek hasil retrieval langsung (SAMA dengan Web App)
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1. Load model + embeddings (identik dengan web)
# ------------------------------------------------------------
print("🔍 Loading model E5...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

data = np.load("embeddings_all.npz", allow_pickle=True)
corpus_emb = data["corpus_embeddings"]
answers    = data["answers"]
questions  = data["questions"]

print("📌 Embeddings loaded:", len(questions), "QA pairs.\n")

# ------------------------------------------------------------
# 2. Retrieval versi terminal (identik dengan web)
# ------------------------------------------------------------
def retrieve_terminal(question):
    pref = "query: " + question          # SAMA dgn web
    q_emb = model.encode(pref, convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    scores = np.dot(corpus_emb, q_emb)   # cosine similarity SAMA dgn web
    idx = scores.argmax()                # ambil TOP-1

    return answers[idx], float(scores[idx])


# ------------------------------------------------------------
# 3. LOOP INPUT MANUAL (untuk menguji pertanyaan apapun)
# ------------------------------------------------------------
while True:
    question = input("Masukkan pertanyaan: ")

    if question.lower().strip() == "exit":
        break

    # Hasil final (persis seperti WEB)
    answer, score = retrieve_terminal(question)

    print("\n🔵 HASIL RETRIEVAL")
    print("Jawaban:", answer)
    print("Skor   :", score)
    print("===============================================")
