# =============================================================
# STREAMLIT APP — Medical QA Retrieval (Final Design)
# =============================================================

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import re

# =============================================================
# 1. Load Model + Embeddings (cached)
# =============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_resource
def load_embeddings():
    data = np.load("embeddings_all.npz", allow_pickle=True)
    return (
        data["corpus_embeddings"],
        data["answers"],
        data["questions"],
    )

model = load_model()
corpus_emb, corpus_ans, corpus_qs = load_embeddings()

# =============================================================
# 2. Retrieval Function
# =============================================================
def retrieve(query):
    q = "query: " + query
    q_emb = model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(corpus_emb, q_emb)
    ranked = scores.argsort()[::-1]
    return ranked, scores

# =============================================================
# 3. Validation Functions
# =============================================================
def is_meaningful_question(text):
    """Check if the question is meaningful"""
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    if len(cleaned) < 5:
        return False
    
    words = cleaned.split()
    meaningful_words = [word for word in words if len(word) >= 3]
    
    if len(meaningful_words) < 1:
        return False
    
    if re.search(r'(.)\1{3,}', cleaned):
        return False
    
    return True

# =============================================================
# 4. Streamlit Config
# =============================================================
st.set_page_config(
    page_title="MediSearch - Konsultasi Medis AI",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Navbar Shadow */
header[data-testid="stHeader"] {
    box-shadow: 0 4px 12px rgba(0,0,0,0.18) !important;
    background-color: white !important;
}

/* Optional: warna teks navbar */
header[data-testid="stHeader"] * {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# 5. Custom CSS - Clean and Modern
# =============================================================
st.markdown("""
<style>
    /* Main Colors */
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #3b82f6;
        --light-bg: #f8fafc;
        --dark-bg: #0f172a;
        --text-light: #1e293b;
        --text-dark: #f1f5f9;
        --card-light: #ffffff;
        --card-dark: #1e293b;
        --border-light: #e2e8f0;
        --border-dark: #334155;
    }

    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #fff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Feature Cards */
    .feature-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        gap: 24px;
    }
    
    .feature-card {
        flex: 1;
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    
    .feature-card p {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
  
    
    /* Input Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
    }
    
    /* Answer Card */
    .answer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .answer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .answer-text {
        color: #374151;
        line-height: 1.7;
        font-size: 1rem;
        text-align: justify;
    }
    
    /* Related Answers - Updated Design */
    .related-answer-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .related-answer-item:hover {
        border-color: #10b981;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .answer-number {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    
    .answer-content {
        flex: 1;
    }
    
    .answer-preview {
        color: #374151;
        line-height: 1.6;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .answer-score {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #dbeafe, #e0f2fe);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7, #fef7cd);
        border-left: 4px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        color: #92400e;
    }

    /* Disclaimer */
    .disclaimer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem;
        color: #64748b;
        background: white;
        border-radius: 16px;
        margin-top: 4rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .feature-container {
            flex-direction: column;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .content-container {
            padding: 1.5rem;
        }
        
        .related-answer-item {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .answer-number {
            align-self: flex-start;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# 6. Initialize Session State
# =============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# =============================================================
# 7. Header Section
# =============================================================
st.markdown("""
<div class="main-header">
    <h1>🩺 MediSearch</h1>
    <p>Temu Kembali Informasi Medis Berbasis Pencarian</p>
    <div style="height: 3px; width: 100px; background: rgba(255,255,255,0.5); border-radius: 2px; margin: 0 auto;"></div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# 9. Main Content Container
# =============================================================
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# =============================================================
# 10. Consultation Section
# =============================================================
st.markdown("### 💬 Konsultasi Medis Anda")

# Tips Box
st.markdown("""
<div class="info-box">
    <h4 style="margin-top: 0; color: #1e40af;">💡 Tips Konsultasi yang Efektif</h4>
    <p style="margin-bottom: 0;">
        • <strong>Jelaskan gejala dengan detail</strong> - Sertakan lokasi, intensitas, dan waktu<br>
        • <strong>Sebutkan durasi keluhan</strong> - Berapa lama gejala berlangsung<br>
        • <strong>Riwayat kesehatan relevan</strong> - Kondisi medis yang pernah dialami<br>
        • <strong>Gunakan bahasa jelas</strong> - Deskripsikan dengan kata-kata yang mudah dipahami
    </p>
</div>
""", unsafe_allow_html=True)

# Input Form
with st.form(key='medical_query_form'):
    question = st.text_area(
        "**Masukkan pertanyaan kesehatan Anda:**",
        height=120,
        placeholder="Contoh: Saya mengalami demam tinggi 39°C selama 2 hari disertai batuk berdahak dan sesak napas ringan. Riwayat asma sejak kecil. Apakah ini gejala COVID-19?",
        help="Jelaskan keluhan kesehatan Anda sejelas mungkin untuk hasil yang akurat"
    )
    
    submit_button = st.form_submit_button("🚀 **Cari Jawaban**", use_container_width=True)

# =============================================================
# 11. Processing and Results
# =============================================================
THRESHOLD = 0.85
TOP_K = 5  # Fixed to 5 related answers

if submit_button:
    cleaned_question = question.strip()
    
    # Validation
    if not cleaned_question:
        st.error("❌ **Pertanyaan kosong** - Silakan masukkan pertanyaan kesehatan Anda")
    elif not is_meaningful_question(cleaned_question):
        st.markdown("""
        <div class="warning-box">
            <h4 style="margin-top: 0; color: #92400e;">🤔 Pertanyaan Tidak Jelas</h4>
            <p style="margin-bottom: 0;">
                Maaf, pertanyaan yang Anda masukkan kurang jelas atau terlalu singkat. 
                Untuk mendapatkan jawaban yang akurat, mohon berikan detail lebih lanjut tentang:
            </p>
            <ul style="margin-bottom: 0;">
                <li>Gejala yang dialami</li>
                <li>Durasi keluhan</li>
                <li>Lokasi dan intensitas gejala</li>
                <li>Riwayat kesehatan yang relevan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick examples
        st.info("""
        **💡 Contoh pertanyaan yang baik:**
        - *"Saya mengalami nyeri dada sebelah kiri yang muncul saat aktivitas fisik, disertai sesak napas. Riwayat hipertensi. Apakah ini berbahaya?"*
        - *"Anak saya usia 3 tahun demam 38.5°C selama 2 hari, disertai batuk dan nafsu makan menurun. Apakah perlu dibawa ke dokter?"*
        - *"Apa perbedaan gejala flu biasa dengan COVID-19? Saya mengalami pilek, sakit tenggorokan, dan sedikit demam."*
        """)
    else:
        # Processing
        with st.spinner("🔍 **Menganalisis pertanyaan dan mencari jawaban terbaik...**"):
            start_time = time.time()
            ranked, scores = retrieve(cleaned_question)
            processing_time = time.time() - start_time

            best_idx = ranked[0]
            best_answer = corpus_ans[best_idx]
            best_score = float(scores[best_idx])

            # === Tambahkan di sini ===
            if best_answer is None or str(best_answer).lower() == "nan":
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin-top: 0; color: #92400e;">🤔 Jawaban Tidak Valid</h4>
                    <p style="margin-bottom: 0;">
                        Maaf, sistem tidak dapat menemukan jawaban yang sesuai untuk pertanyaan Anda. 
                        Hal ini dapat terjadi jika data tidak lengkap atau tidak sesuai.
                        Mohon berikan detail lebih lanjut tentang:
                    </p>
                    <ul style="margin-bottom: 0%;">
                        <li>Gejala yang dialami</li>
                        <li>Durasi keluhan</li>
                        <li>Lokasi dan intensitas gejala</li>
                        <li>Riwayat kesehatan yang relevan</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            # =========================


        # Get relevant candidates (excluding the main answer)
        candidates = [
            {
                "answer": corpus_ans[i],
                "score": float(scores[i])
            }
            for i in ranked[1:TOP_K+1]  # Skip the first one (main answer)
            if float(scores[i]) >= THRESHOLD
        ]

        # Main Answer
        st.markdown("---")
        st.markdown("### 🎯 **Jawaban Utama**")
        
        # Score styling
        if best_score >= 0.9:
            score_emoji = "✅"
            score_text = "Sangat Relevan"
        elif best_score >= 0.85:
            score_emoji = "ℹ️"
            score_text = "Relevan"
        else:
            score_emoji = "⚠️"
            score_text = "Cukup Relevan"

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-header">
                <div>
                    <span class="score-badge">
                        {score_emoji} {score_text} • Skor: {best_score:.4f}
                    </span>
                </div>
                <div style="color: #64748b; font-size: 0.9rem;">
                    ⏱️ Diproses dalam {processing_time:.2f} detik
                </div>
            </div>
            <div class="answer-text">
                {best_answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Low score warning
        if best_score < THRESHOLD:
            st.warning("""
            **Perhatian**: Jawaban ini memiliki tingkat relevansi yang sedang. 
            Untuk diagnosis dan penanganan yang tepat, disarankan untuk berkonsultasi 
            langsung dengan dokter atau tenaga medis profesional.
            """)

        # =============================================================
        # Related Answers - Updated Design (tanpa duplikat & tanpa nan)
        # =============================================================
        if True:  # dipaksa True agar tidak NameError
            if candidates:
                st.markdown("---")
                st.markdown("### 💡 **Jawaban lain yang mungkin membantu:**")

                unique_candidates = []
                seen_texts = []

                def is_duplicate(a, b, threshold=0.88):
                    """Cek kemiripan sederhana untuk hilangkan duplikat."""
                    a_norm = a.replace(" ", "").lower()
                    b_norm = b.replace(" ", "").lower()
                    if not a_norm or not b_norm:
                        return False
                    same = sum(1 for x, y in zip(a_norm, b_norm) if x == y)
                    sim = same / max(len(a_norm), len(b_norm))
                    return sim >= threshold

                # Filter duplikat & nan
                for item in candidates:
                    text = str(item["answer"]).strip()

                    # Skip nan / kosong
                    if not text or text.lower() == "nan":
                        continue

                    # Skip kalau mirip dengan jawaban utama
                    if is_duplicate(text, str(best_answer)):
                        continue

                    # Skip kalau mirip dengan candidate yang sudah masuk
                    if any(is_duplicate(text, t) for t in seen_texts):
                        continue

                    unique_candidates.append(item)
                    seen_texts.append(text)

                # Tampilkan candidate unik
                for idx, candidate in enumerate(unique_candidates, start=1):
                    preview_text = str(candidate["answer"]).strip()

                    if not preview_text or preview_text.lower() == "nan":
                        continue

                    # BUAT PREVIEW AGAR RAPI
                    if len(preview_text) > 100:
                        snippet = preview_text[:100]
                        if "." in snippet:
                            cutoff = snippet.rfind(".") + 1
                        elif " " in snippet:
                            cutoff = snippet.rfind(" ") + 1
                        else:
                            cutoff = 100
                        preview_text = preview_text[:cutoff].strip() + "..."

                    st.markdown(f"""
                    <div class="related-answer-item">
                        <div class="answer-number">{idx}</div>
                        <div class="answer-content">
                            <div class="answer-preview">{preview_text}</div>
                            <div class="answer-score">Skor {candidate['score']:.4f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# Close content container
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================
# 8. Features Section
# =============================================================
st.markdown("""
<div class="feature-container">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h3>Pencarian Cerdas</h3>
        <p>search engine medis berbasis kesesuaian data</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Hasil Instan</h3>
        <p>Dapatkan jawaban relevan dalam hitungan detik tanpa menunggu lama</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <h3>Informasi Relevan</h3>
        <p>Informasi berasal dari dataset medis yang relevan</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================
# 12. Footer dengan margin yang cukup
# =====================
# ===================
# =====================
st.markdown("""
<div class="footer">
    <h3 style="color: #1e293b; margin-bottom: 1rem;">🏥 MediSearch</h3>
    <p style="margin-bottom: 0.5rem; font-size: 1.1rem;">
        <strong>Sistem Temu Kembali Informasi Medis Berbasis AI</strong>
    </p>
    <p style="margin: 0; font-size: 0.9rem; line-height: 1.5; max-width: 600px; margin: 0 auto;">
        <strong>Disclaimer Medis:</strong> Informasi yang diberikan bersifat edukatif dan informatif semata. 
        Tidak menggantikan konsultasi langsung dengan dokter atau tenaga medis profesional. 
        Selalu konsultasikan masalah kesehatan dengan ahli yang kompeten.
    </p>
</div>
""", unsafe_allow_html=True)