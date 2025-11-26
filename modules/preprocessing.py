# ============================================================
# 🧹 Pra-pemrosesan Dataset Medis untuk SBERT
# (FINAL – Question aman, Answer agresif + tanpa hapus nama +
#  hapus kalimat penutup + NO DATA LOSS)
# ============================================================

import pandas as pd
import re
import os
import time

start_time = time.time()
print("🚀 Memulai proses pra-pemrosesan dataset...\n")

# ------------------------------------------------------------
# 1) Baca dataset
# ------------------------------------------------------------
DATA_IN  = "DATASET TANYA JAWAB MEDIS.xlsx"
DATA_OUT = "DATASET TANYA JAWAB CLEAN_QA.xlsx"

if not os.path.exists(DATA_IN):
    raise FileNotFoundError(f"File '{DATA_IN}' tidak ditemukan!")

df = pd.read_excel(DATA_IN)
print(f"✅ Dataset dibaca. Total baris: {len(df)}\n")

if "question" not in df.columns or "answer" not in df.columns:
    raise ValueError("Kolom 'question' dan 'answer' wajib ada!")

df = df[["question", "answer"]].dropna().copy()
print(f"📊 Kolom penting dipilih. Total valid: {len(df)}\n")


# ============================================================
#   📌 BAGIAN YANG SAMA UNTUK KEDUA VERSI
# ============================================================
def tidy_punct(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("/", " ")
    text = re.sub(r"[\r\n]+", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", text)
    return text.strip()


# ============================================================
# INTRO PATTERN
# ============================================================
intro_keyword_patterns = [
    r"^\s*(assalamualaikum|assalamu'alaikum|waalaikumsalam)\b",
    r"^\s*(halo|hai|alo|permisi|selamat\s+(pagi|siang|sore|malam))\b",
    r"^\s*(dok|dokter)\b",
    r"^\s*(nama\s*saya|perkenalkan)\b",
    r"^\s*(saya)\s+(ingin|mau|akan|ingin\s*bertanya|mau\s*bertanya)\b",
    r"^\s*(mohon|minta|tolong)\b",
    r"^\s*(terima\s*kasih|makasih)\b",
    r"^\s*(untuk\s*pertanyaan\s*anda|berdasarkan\s*pertanyaan\s*anda|menjawab\s*pertanyaan)\b",
    r"^\s*(sebelumnya\s*terima\s*kasih|sebelumnya\s*maaf)\b",
    r"^\s*(pertanyaan\s*anda|anda\s*bertanya)\b",
    r"^\s*(salam(?:\s+hormat|\s+sehat)?)\b",
    r"^\s*(bertanya|saya\s*bertanya)\b",
    r"^\s*(terkait|mengenai)\s+(pertanyaan|keluhan)\b",
    r"^\s*(dok(?:ter)?[:,]?\s*(saya|mau|ingin)?)\b",
]
intro_regexes = [re.compile(p, re.IGNORECASE) for p in intro_keyword_patterns]


# ============================================================
# 🔹 Hapus kalimat penutup (closing)
# ============================================================
def remove_closing_statements(text: str) -> str:
    closing_patterns = [
        r"demikian[^.]*$",
        r"semoga (membantu|bermanfaat)[^.]*$",
        r"terima kasih[^.]*$",
        r"sekian[^.]*$",
        r"salam sehat[^.]*$"
    ]
    for p in closing_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
    return text


# ============================================================
#  🟦 CLEAN QUESTION  (VERSI AMAN)
# ============================================================
def remove_leading_intro_sentences_question(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if text == "":
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) == 1:
        return tidy_punct(text)

    first = sentences[0].strip()
    if "?" in first:
        return tidy_punct(text)

    first_word_count = len(re.findall(r"\w+", first))
    looks_like_intro = False

    for rgx in intro_regexes:
        if rgx.match(first) or rgx.search(first):
            looks_like_intro = True
            break

    if not looks_like_intro or first_word_count > 10:
        return tidy_punct(text)

    remaining = sentences[1:]
    result = " ".join([s.strip() for s in remaining if s.strip() != ""])
    if not result:
        return tidy_punct(text)
    return tidy_punct(result)


def clean_question_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", ". ").replace("\n", ". ")
    text = text.replace("/", " ")
    text = text.strip()
    text = remove_leading_intro_sentences_question(text)
    text = re.sub(r"[^a-zA-Z0-9À-ÿ\s\.,;:!?%()\-\']", " ", text)
    text = tidy_punct(text)
    return text.lower().strip()


# ============================================================
#  🟩 CLEAN ANSWER (VERSI AGRESIF + TANPA HAPUS NAMA)
# ============================================================
def remove_leading_intro_sentences_answer(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if text == "":
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    remaining = []
    skip_mode = True

    for sent in sentences:
        s = sent.strip()
        if s == "":
            continue

        is_intro = False
        words = re.findall(r"\w+", s)

        if len(words) <= 6:
            for rgx in intro_regexes:
                if rgx.search(s):
                    is_intro = True
                    break
        else:
            for rgx in intro_regexes:
                if rgx.match(s):
                    is_intro = True
                    break

        if skip_mode and is_intro:
            continue
        else:
            skip_mode = False
            remaining.append(s)

    if not remaining:
        return tidy_punct(text)

    result = " ".join(remaining).strip()
    return tidy_punct(result)


def clean_answer_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", ". ").replace("\n", ". ")
    text = text.replace("/", " ")

    text = remove_leading_intro_sentences_answer(text)

    text = re.sub(r"\b(di|kepada|pada)\s*alodokter\b", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9À-ÿ\s\.,;:!?%()\-\']", " ", text)

    # 🔥 Hapus kalimat penutup
    text = remove_closing_statements(text)

    text = tidy_punct(text)
    return text.lower().strip()


# ============================================================
# Terapkan Kedua Cleaners (TANPA FILTER BARIS)
# ============================================================
df["clean_question"] = df["question"].apply(clean_question_text)
df["clean_answer"]   = df["answer"].apply(clean_answer_text)

print(f"✨ Pembersihan selesai. Total baris tetap: {len(df)}\n")

# ============================================================
# OUTPUT
# ============================================================
out = df[["clean_question", "clean_answer"]].rename(
    columns={
        "clean_question": "question",
        "clean_answer":   "answer"
    }
)

out.to_excel(DATA_OUT, index=False)
print(f"📁 Disimpan: {DATA_OUT}")
print(f"\n⏱ Waktu total: {round(time.time()-start_time, 2)} detik")
