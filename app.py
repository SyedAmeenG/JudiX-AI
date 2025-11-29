# app_full_mvp.py
"""
Full MVP single-file Streamlit app
JudiX – AI Judicial Justice Companion
- Virtual Courtroom simulator
- Multilingual Chatbot (Gemini + retrieval over judiciary sites)
- Document Understander
- Judicial Search with citations
"""

import os, time, json, textwrap, tempfile
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langdetect import detect
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
import pdfplumber
import pytesseract

load_dotenv()

# ---------------- CONFIG ----------------
LOGO_PATH = "JudiX Logo.png"   # logo file in your project folder
SEED_URLS = [
    "https://ecourts.gov.in",
    "https://njdg.ecourts.gov.in",
    "https://main.sci.gov.in",
    "https://legalaffairs.gov.in",
    "https://indiacode.nic.in",
]
CHUNK_SIZE = 1500
TOP_K_DEFAULT = 6
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
USER_AGENT = "judicial-mvp/1.0 (+https://example.org)"

# ----------------- UI style -----------------
st.set_page_config(
    page_title="JudiX – AI Judicial Justice Companion",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #000000 100%);
            color: #e6eef8;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }
        .big-title {
            font-size: 32px;
            font-weight: 800;
            color: #ffffff;
        }
        .subtitle {
            font-size: 15px;
            color: #cbd5f5;
            margin-top: 4px;
        }
        .mode-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 500;
            margin-top: 6px;
            background: rgba(148, 163, 184, 0.18);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .card {
            background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(15,23,42,0.80));
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.22);
        }
        .sidebar .stButton>button, .stButton>button {
            border-radius: 999px !important;
        }
        /* ---------- TABS: equal width, centered text ---------- */
        .stTabs [data-baseweb="tab-list"] {
            display: flex !important;
            gap: 10px;
            border-bottom: 4px solid rgba(148,163,184,0.2);
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1 !important;                     /* all tabs same width */
            padding-top: 6px;
            padding-bottom: 6px;
            border-radius: 999px;
            background-color: rgba(15,23,42,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] p {
            margin: 0;
            width: 100%;
            text-align: center;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;                /* if text is too long */
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(120deg, #4f46e5, #38bdf8) !important;
            color: white !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] p {
            color: #ffffff !important;
        }
        .chat-bot {
            background: #020617;
            border: 1px solid #374151;
            color: #e5e7eb;
            padding: 9px 13px;
            border-radius: 18px;
            margin: 6px 0;
            max-width: 78%;
            font-size: 14px;
        }
        .chat-meta {
            font-size: 10px;
            color: #9ca3af;
            margin-top: 3px;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
mode_label = "Live (Gemini enabled)" if GEMINI_API_KEY else "Offline Demo"

col_logo, col_title = st.columns([1, 9])
with col_logo:
    try:
        logo_img = Image.open(LOGO_PATH)
        st.image(logo_img, width=110)
    except Exception:
        st.write("⚖️")

with col_title:
    st.markdown('<div class="big-title">JudiX – AI Judicial Justice Companion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Virtual Courtroom · Multilingual Legal Tutor · Document Explainability · Judiciary-Only Search with Citations</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="mode-pill">Current mode: {mode_label}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Prototype Controls")
mode = st.sidebar.radio("Mode", ["Live (Gemini)", "Offline Demo"])
st.sidebar.caption("Set GEMINI_API_KEY in environment or .env for live Gemini responses.")
st.sidebar.markdown("**Seed judiciary sites** (edit in *Judicial Search* tab):")
for u in SEED_URLS:
    st.sidebar.write("• ", u)

# No explicit Gemini “detected” acknowledgement – sidebar kept clean.

# button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state["chat_history"] = []

# ---------------- Utilities ----------------
def fetch_page_text(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script", "style", "noscript", "iframe", "svg"]):
            s.decompose()
        texts = []
        for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                texts.append(txt)
        return url, "\n".join(texts)
    except Exception:
        return url, ""

def crawl_seed_urls(seed_urls, max_pages=40, per_seed_limit=5):
    visited, pages = set(), []
    for seed in seed_urls:
        try:
            base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(seed))
            r = requests.get(seed, timeout=10, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            links = {urljoin(base, a.get('href')) for a in soup.find_all('a', href=True)}
            candidates = [seed] + [l for l in links if l.startswith(base)]
            cnt = 0
            for link in candidates:
                if cnt >= per_seed_limit: break
                if link in visited: continue
                visited.add(link)
                u, text = fetch_page_text(link)
                if text and len(text) > 200:
                    pages.append({"url": u, "text": text})
                    cnt += 1
        except Exception:
            continue
        if len(pages) >= max_pages: break
    return pages

def chunk_text(text, approx_chars=CHUNK_SIZE):
    text = text.replace("\r", " ").replace("\n", " ")
    tokens = text.split()
    approx_words = max(50, approx_chars // 5)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+approx_words])
        chunks.append(chunk)
        i += approx_words
    return chunks

def build_corpus(pages):
    chunk_texts, metas = [], []
    for p in pages:
        url = p["url"]
        for c in chunk_text(p["text"]):
            if len(c.strip()) < 50: continue
            chunk_texts.append(c)
            metas.append({"url": url, "snippet": c[:300] + ("..." if len(c) > 300 else "")})
    return chunk_texts, metas

def build_tfidf_index(chunk_texts):
    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vec.fit_transform(chunk_texts)
    return vec, X

def retrieve_top_k(query, vec, X, chunk_texts, metas, k=TOP_K_DEFAULT):
    qv = vec.transform([query])
    sims = linear_kernel(qv, X).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    results = [{"score": float(sims[idx]), "text": chunk_texts[idx], "meta": metas[idx]} for idx in top_idx]
    return results

def call_gemini_with_context(question, contexts, model_name=GEMINI_MODEL):
    context_blocks = []
    for i, c in enumerate(contexts):
        context_blocks.append(f"[SRC {i+1}] URL: {c['meta']['url']}\nSNIPPET: {c['meta']['snippet']}\nTEXT: {c['text']}\n")
    context_str = "\n\n".join(context_blocks)
    prompt = (
        "You are an assistant that MUST answer using ONLY the provided sources below. "
        "Do NOT hallucinate. If not present, say you cannot find an authoritative answer in the provided sources.\n\n"
        f"Sources:\n{context_str}\n\n"
        "INSTRUCTIONS:\n1) Provide a concise answer (2-6 sentences).\n"
        "2) After the answer, list source indices and URLs you used.\n"
        "3) If you quote, include short quoted snippets referencing the source index.\n\n"
        f"User question: {question}\n\nAnswer:"
    )

    try:
        import google.generativeai as genai
        if GEMINI_API_KEY and mode.startswith("Live"):
            genai.configure(api_key=GEMINI_API_KEY)
            resp = genai.generate_text(
                model=model_name, prompt=prompt, temperature=0.0, max_output_tokens=512
            )
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
        else:
            raise RuntimeError("No Gemini key / not in Live mode")
    except Exception:
        out = "DEMO SYNTHESIS — based only on top sources:\n\n"
        for i, c in enumerate(contexts[:4]):
            out += f"[SRC {i+1}] {c['meta']['url']}\n  Snippet: \"{c['meta']['snippet'][:200]}\"\n\n"
        out += "Enable GEMINI_API_KEY and google.generativeai for live constrained answers."
        return out

def detect_lang(text):
    try:
        return detect(text)
    except Exception:
        return "en"

def extract_text_from_pdfbytes(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            text = ""
            with pdfplumber.open(tmp.name) as pdf:
                for p in pdf.pages:
                    text += p.extract_text() or ""
            return text
    except Exception:
        try:
            img = Image.open(tempfile.NamedTemporaryFile(delete=False).name)
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

def summarize_document_via_demo(doc_text):
    s = doc_text.strip()
    summary = s[:400] + ("..." if len(s) > 400 else "")
    deadlines = []
    import re
    days = re.findall(r"(\d{1,3}\s+days)", s.lower())
    if days:
        deadlines = days[:3]
    return {
        "summary": summary,
        "deadlines": deadlines,
        "actions": [
            "Read notice carefully",
            "Contact legal aid / lawyer",
            "Do not miss deadlines",
        ],
        "risk": "Medium (demo)",
    }

# ----------------- Prepare retrieval index (cached) -----------------
@st.cache_data(ttl=60 * 60)
def prepare_index(seed_urls):
    with st.spinner("Crawling seed sites & building index (shallow) ..."):
        pages = crawl_seed_urls(seed_urls, max_pages=40)
        chunk_texts, metas = build_corpus(pages)
        if not chunk_texts:
            return None
        vec, X = build_tfidf_index(chunk_texts)
        return {"vec": vec, "X": X, "chunks": chunk_texts, "metas": metas}

# ---------------- Main app layout (tabs) ----------------
tabs = st.tabs(
    [
        "Overview",
        "Virtual Courtroom",
        "Chatbot",
        "Document Understander",
        "Judicial Search",
        "Demo Script & Notes",
    ]
)

# ---------- Tab: Overview ----------
with tabs[0]:
    st.subheader("What JudiX demonstrates")
    st.markdown(
        textwrap.dedent(
            """
        - An immersive **Virtual Courtroom** to practice what happens in common hearings.
        - A **Multilingual Legal Navigator Chatbot** (retrieval-first, Gemini-enabled).
        - A **Document Understander** to simplify legal notices, FIRs, and summons.
        - **Judicial Search**: answers backed only by official judiciary sites, with explicit citations.
        """
        )
    )
    st.info(
        "Current mode: "
        + (
            "Live (Gemini enabled)"
            if GEMINI_API_KEY and mode.startswith("Live")
            else "Offline Demo (no Gemini key)"
        )
    )
    st.markdown("---")
    st.markdown(
        "🔎 **Quick path for judges/mentors**: Virtual Courtroom → Chatbot → Document Understander → Judicial Search."
    )

# ---------- Tab: Virtual Courtroom ----------
with tabs[1]:
    st.header("Virtual Courtroom Simulator — Roleplay & Practice")
    st.markdown(
        "Scenario templates are short, realistic, and designed to reduce fear before entering a real courtroom."
    )
    scenario = st.selectbox(
        "Choose a scenario",
        [
            "Consumer complaint — faulty product (recommended)",
            "FIR filing — theft/damage",
            "Domestic dispute — emergency orders (informational)",
        ],
    )
    role = st.selectbox(
        "Select your role",
        ["Complainant / Petitioner", "Respondent", "Witness", "Observer"],
    )

    if "vc_step" not in st.session_state:
        st.session_state["vc_step"] = 0

    def next_vc():
        st.session_state["vc_step"] = st.session_state.get("vc_step", 0) + 1

    def prev_vc():
        st.session_state["vc_step"] = max(0, st.session_state.get("vc_step", 0) - 1)

    nav_cols = st.columns([1, 1, 1, 6])
    with nav_cols[0]:
        if st.button("◀ Previous", key="vc_prev"):
            prev_vc()
    with nav_cols[1]:
        if st.button("Next ▶", key="vc_next"):
            next_vc()
    with nav_cols[2]:
        if st.button("Reset ⟳", key="vc_reset"):
            st.session_state["vc_step"] = 0

    steps = [
        {
            "title": "Intro",
            "text": "You arrive at the district consumer forum. You have a bill, product photos, and communication records with the seller.",
        },
        {
            "title": "Filing",
            "text": "You file the complaint — attach evidence and clearly state your claim amount and what remedy you want (refund/replacement/compensation).",
        },
        {
            "title": "Service of Notice",
            "text": "The court issues a notice to the seller. They must respond within the time period given in the notice.",
        },
        {
            "title": "First Hearing",
            "text": "The judge may ask basic questions or invite both parties to mediation. Stay calm and focus on facts and dates.",
        },
        {
            "title": "Evidence & Witness",
            "text": "You may be asked to show documents and bring witnesses. You answer questions truthfully and briefly.",
        },
        {
            "title": "Orders & Execution",
            "text": "If the court decides in your favour, it may order refund, replacement, or compensation. If the seller does not comply, you can file for execution of the order.",
        },
    ]
    idx = min(st.session_state["vc_step"], len(steps) - 1)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(steps[idx]["title"])
    st.write(steps[idx]["text"])

    if idx == 1:
        st.info(
            "📎 Checklist for filing: Bill, Photos, Warranty Card, Serial number, Communication logs, Purchase invoice."
        )
        upload = st.file_uploader(
            "Upload a sample bill/photo (optional – for demo only)",
            type=["png", "jpg", "pdf"],
            key="vc_evidence",
        )
        if upload:
            st.success("Uploaded. In a full product, this would be auto-tagged as evidence.")
    if idx == 3:
        st.write("🗣️ Practice: The judge asks *“When did you buy this product?”*")
        ans = st.text_input("Type your answer as you would say it in court:", key="vc_statement")
        if st.button("Check my answer", key="vc_practice"):
            st.success("Nice. Keep it factual: date, place of purchase, and refer to your bill if needed.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("This simulator is for learning only and does not replace real legal guidance.")

# ---------- Tab: Chatbot ----------
with tabs[2]:
    st.header("Multilingual Legal Navigator Chatbot")
    st.markdown(
        "Ask common legal questions in English, Hindi, or other Indian languages. "
        "Responses are grounded in judiciary sources where possible and end with a disclaimer."
    )

    # initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_cols = st.columns([8, 2])
    with chat_cols[0]:
        user_q = st.text_input(
            "Ask your question (e.g., 'Mujhe legal notice mila, kya karu?')",
            key="chat_input",
        )
    with chat_cols[1]:
        resp_lang = st.selectbox(
            "Reply language", ["en", "hi", "ta", "te", "kn"], index=0
        )

    send_clicked = st.button("Send", key="chat_send")

    if send_clicked:
        if not user_q.strip():
            st.warning("Please type a question.")
        else:
            detected = detect_lang(user_q) if resp_lang == "auto" else resp_lang
            # retrieval grounding
            index_data = prepare_index(st.session_state.get("_seed_urls", SEED_URLS))
            top = []
            if index_data:
                top = retrieve_top_k(
                    user_q,
                    index_data["vec"],
                    index_data["X"],
                    index_data["chunks"],
                    index_data["metas"],
                    k=4,
                )

            reply = call_gemini_with_context(user_q, top[:4] if top else [])
            timestamp = time.strftime("%H:%M")

            st.session_state["chat_history"].append(
                {"role": "user", "text": user_q, "time": timestamp, "lang": detected}
            )
            st.session_state["chat_history"].append(
                {"role": "bot", "text": reply, "time": timestamp, "lang": "bot"}
            )

    # show chat history with custom bubbles
    for i, msg in enumerate(st.session_state["chat_history"][-10:]):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-user">
                    {msg['text']}
                    <div class="chat-meta">{msg['time']} · You</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-bot">
                    {msg['text']}
                    <div class="chat-meta">{msg['time']} · JudiX assistant</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("Educational guidance only — not a substitute for a licensed advocate.")

# ---------- Tab: Document Understander ----------
with tabs[3]:
    st.header("Document Understander — Legal Notice / FIR / Summons")
    st.markdown(
        "Upload a document and JudiX will try to summarise it in plain language, highlight deadlines, and suggest immediate actions."
    )

    uploaded = st.file_uploader(
        "Upload document (pdf, txt, png, jpg)", type=["pdf", "txt", "png", "jpg", "jpeg"]
    )
    pasted = st.text_area(
        "Or paste the document text here (optional)", height=140
    )

    if st.button("Analyze Document", key="analyze_doc"):
        doc_text = ""
        if uploaded:
            bt = uploaded.read()
            if uploaded.type == "application/pdf":
                doc_text = extract_text_from_pdfbytes(bt)
            else:
                try:
                    doc_text = bt.decode("utf-8")
                except Exception:
                    try:
                        img = Image.open(uploaded)
                        doc_text = pytesseract.image_to_string(img)
                    except Exception:
                        doc_text = ""
        if pasted and not doc_text.strip():
            doc_text = pasted

        if not doc_text.strip():
            st.error("No document text found. Please upload or paste content.")
        else:
            st.info("Analyzing document...")
            if GEMINI_API_KEY and mode.startswith("Live"):
                contexts = [{"meta": {"url": "uploaded-doc"}, "text": doc_text}]
                analysis = call_gemini_with_context(
                    "Summarize this document and extract deadlines and immediate actions clearly.",
                    contexts,
                )
                st.subheader("AI Summary & Extracted Items")
                st.write(analysis)
            else:
                res = summarize_document_via_demo(doc_text)
                st.subheader("Summary (demo)")
                st.write(res["summary"])
                st.subheader("Deadlines")
                for d in res["deadlines"]:
                    st.write("- " + str(d))
                st.subheader("Immediate actions")
                for a in res["actions"]:
                    st.write("- " + a)
                st.subheader("Risk rating")
                st.write(res["risk"])

    st.markdown("---")
    st.caption("Please avoid uploading sensitive personal details. This is an educational triage tool.")

# ---------- Tab: Judicial Search ----------
with tabs[4]:
    st.header("Judicial Search — Retrieval-First Answers with Citations")
    st.markdown(
        "JudiX searches a curated list of official judiciary and law sites, retrieves the most relevant sections, "
        "and then asks Gemini to answer **only** from those excerpts."
    )

    with st.expander("Edit seed URLs"):
        seed_text = st.text_area(
            "One URL per line",
            value="\n".join(st.session_state.get("_seed_urls", SEED_URLS)),
            height=180,
        )
        if st.button("Update Index", key="update_seeds"):
            new_seeds = [s.strip() for s in seed_text.splitlines() if s.strip()]
            st.session_state["_seed_urls"] = new_seeds
            st.success("Seed list updated. Rebuilding index...")
            st.cache_data.clear()
            st.experimental_rerun()

    index_data = prepare_index(st.session_state.get("_seed_urls", SEED_URLS))
    if not index_data:
        st.warning("Index not ready. Please check seed URLs or your network.")
    q = st.text_input(
        "Ask a process-focused question (e.g., 'How to check case status on eCourts?')",
        key="search_q",
    )
    search_cols = st.columns([3, 1])
    with search_cols[1]:
        k = st.number_input(
            "Top-K chunks", min_value=1, max_value=12, value=TOP_K_DEFAULT
        )
        if st.button("Rebuild index now", key="rebuild_index"):
            st.cache_data.clear()
            st.experimental_rerun()

    if q and index_data:
        with st.spinner("Retrieving top sources from judiciary sites..."):
            top = retrieve_top_k(
                q, index_data["vec"], index_data["X"], index_data["chunks"], index_data["metas"], k=k
            )
        st.markdown("#### Retrieved snippets (ranked)")
        for i, r in enumerate(top, 1):
            st.markdown(f"**#{i} (score {r['score']:.4f})**")
            st.write(r["meta"]["snippet"])
            st.write(r["meta"]["url"])
            st.divider()

        with st.spinner("Generating answer based only on these snippets..."):
            ans = call_gemini_with_context(q, top[:k])
        st.markdown("### Answer (from cited sources only)")
        st.write(ans)

        st.markdown("### Source Panel")
        seen = set()
        for i, r in enumerate(top, 1):
            url = r["meta"]["url"]
            if url in seen:
                continue
            seen.add(url)
            st.write(f"- {url}")
            st.write(f"  > Snippet: {r['meta']['snippet'][:280]}")

    st.markdown("---")
    st.caption("This search synthesizes public judiciary information and is not legal advice.")

# ---------- Tab: Demo Script & Notes ----------
with tabs[5]:
    st.header("Demo Script & Next Steps")
    st.markdown(
        """
      **Suggested 60–90s demo flow**
      1. Show the JudiX header and describe the problem: judicial illiteracy and fear of courts.
      2. Go to **Virtual Courtroom**, pick the consumer complaint scenario, and practice one judge question.
      3. Move to **Chatbot**, ask a Hindi or English question about a legal notice, and show the structured answer.
      4. Open **Document Understander**, paste a short legal notice, and show summary + deadlines + actions.
      5. Finish with **Judicial Search** – ask how to check case status and highlight that the answer is only from official sites with citations.
      """
    )
    st.subheader("Next steps towards production")
    st.markdown(
        """
      - Replace TF-IDF with embeddings + a vector database (Chroma / FAISS).
      - Respect `robots.txt` and prefer official APIs for judiciary data.
      - Add user accounts, consent flows, and encrypted storage for documents.
      - Provide WhatsApp / IVR access for low-literacy or offline users.
      """
    )
    st.markdown("---")
    st.caption(
        "Footer: Educational prototype (not legal advice). For any real dispute, citizens should consult a lawyer or official legal aid."
    )

# ---------------- Footer / Disclaimer ----------------
st.markdown("---")
st.caption(
    "JudiX is an educational AI companion and does not create a lawyer–client relationship. "
    "Do not upload highly sensitive personal information in this demo."
)
