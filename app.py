# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# ----- Streamlit UI -----
st.set_page_config(page_title="🌙 Umeeda — Assistant", layout="centered")
st.title("🌙 Umeeda — Your Secret Friend")
st.caption("Umeeda answers using your uploaded knowledge base. Admin: use sidebar to upload or reindex.")
# Import KB helpers (make sure kb_loader.py defines these)
from kb_loader import load_index, retrieve, ingest_csv, build_index

# ----- Config / guardrail prompt -----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env. Add it and restart.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
# Use the working model you discovered earlier
model = genai.GenerativeModel("models/gemini-2.5-flash")

CULTURE_SYSTEM_PROMPT = """
You are Umeeda, an assistant for users in Pakistan. Answer respectfully and according to Islamic and local cultural norms.
Follow these rules:
1. Use modest, family-friendly and respectful language.
2. Avoid sexual/explicit content or instructions for harmful/illegal activities.
3. Do not produce content that insults, dehumanizes, or encourages violence towards any person or group.
4. If the user requests something that violates the rules above, politely refuse and optionally provide a safe alternative.
5. Be concise and, when appropriate, reference Islamic values respectfully.
"""

# ----- Load index (if available) -----
INDEX = None
METADATA = None
try:
    INDEX, METADATA = load_index("data/kb_index.faiss", "data/metadata.pkl")
    st.sidebar.success("Knowledge index loaded.")
except Exception:
    INDEX, METADATA = None, None
    # Quiet: index might not exist yet; admin UI can build it.

# ----- Decision logic: short answer vs RAG -----
HIGH_CONFIDENCE_THRESHOLD = 0.78  # tune if needed

def decide_reply(query: str):
    """Return an appropriate reply string using the KB if available."""
    if INDEX is None or METADATA is None:
        # no KB: fallback to simple model call with culture prompt
        prompt = CULTURE_SYSTEM_PROMPT + "\n\nUser: " + query
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                text = getattr(resp.candidates[0], "content", None)
            return text.strip() if text else "Sorry — I couldn't produce an answer."
        except Exception as e:
            return f"Error contacting model: {e}"

    # Use KB retrieval
    results = retrieve(query, INDEX, METADATA, top_k=4)
    if not results:
        return "I couldn't find a direct reference. Please rephrase or ask for more details."

    top = results[0]
    score = float(top.get("score", 0.0))

    # If metadata uses different keys (from CSV ingestion), handle gracefully:
    short_answer = top.get("short_answer") or top.get("text") or ""
    detailed_answer = top.get("detailed_answer") or top.get("text") or ""
    entry_id = top.get("id") or top.get("chunk_id") or "unknown"
    source = top.get("source", "unknown")
    risk = top.get("risk_level", top.get("risk", "Info"))

    # If high confidence, return short answer + citation
    if score >= HIGH_CONFIDENCE_THRESHOLD and short_answer:
        prefix = ""
        # if any retrieved item is urgent, include warning
        if any((r.get("risk_level","").lower()=="urgent") or (r.get("risk","").lower()=="urgent") for r in results):
            prefix = "⚠️ This appears urgent. Please consult a qualified person.\n\n"
        return f"{prefix}{short_answer}\n\nSource: [{source} | ID:{entry_id}]"

    # Otherwise build a RAG prompt using detailed answers
    rag_context = ""
    for r in results:
        r_src = r.get("source", "unknown")
        r_id = r.get("id", r.get("chunk_id", "unknown"))
        r_risk = r.get("risk_level", r.get("risk", "Info"))
        r_detail = r.get("detailed_answer") or r.get("text") or ""
        rag_context += f"[{r_src} | ID:{r_id} | Risk:{r_risk}]\n{r_detail}\n\n"

    prompt = CULTURE_SYSTEM_PROMPT + "\n\nRetrieved knowledge:\n" + rag_context + "\nUser question:\n" + query + "\n\nAnswer using the retrieved knowledge and cite sources."
    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            text = getattr(resp.candidates[0], "content", None)
        final = text.strip() if text else "Sorry — I couldn't produce an answer."
        # Append simple sources list
        sources = ", ".join(sorted({f"{r.get('source','unknown')} (ID {r.get('id', r.get('chunk_id','?'))})" for r in results}))
        return final + "\n\nSources: " + sources
    except Exception as e:
        return f"Error contacting model: {e}"



if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar admin: upload PDFs or import CSV and rebuild
st.sidebar.header("Umeeda — Admin")

# Upload PDF(s)
uploaded_pdf = st.sidebar.file_uploader("Upload PDF to KB (then click Rebuild)", type=["pdf"])
if uploaded_pdf:
    os.makedirs("data/sources", exist_ok=True)
    save_path = os.path.join("data/sources", uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success(f"Saved {uploaded_pdf.name} to data/sources. Click Rebuild KB index to index it.")

# Upload CSV (structured KB)
uploaded_csv = st.sidebar.file_uploader("Upload CSV KB (id,theme,sample_questions,short_answer,detailed_answer,risk_level,source)", type=["csv"])
if uploaded_csv:
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "kb_import.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_csv.getbuffer())
    st.sidebar.success(f"Saved CSV as data/kb_import.csv. Click Import CSV to ingest.")

# Buttons to import CSV or rebuild index
if st.sidebar.button("Import CSV (build index from CSV)"):
    csv_path = os.path.join("data", "kb_import.csv")
    if not os.path.exists(csv_path):
        st.sidebar.error("data/kb_import.csv not found. Upload CSV first.")
    else:
        try:
            st.sidebar.info("Ingesting CSV and building index (this can take a moment)...")
            ingest_csv(csv_path, index_path="data/kb_index.faiss", meta_path="data/metadata.pkl")
            INDEX, METADATA = load_index("data/kb_index.faiss", "data/metadata.pkl")
            st.sidebar.success("CSV ingested and index built.")
        except Exception as e:
            st.sidebar.error(f"Ingest failed: {e}")

if st.sidebar.button("Rebuild KB index from PDFs (data/sources)"):
    try:
        st.sidebar.info("Building index from PDFs in data/sources...")
        build_index("data/sources", index_path="data/kb_index.faiss", meta_path="data/metadata.pkl")
        INDEX, METADATA = load_index("data/kb_index.faiss", "data/metadata.pkl")
        st.sidebar.success("Index rebuilt from PDFs.")
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

# ----- Chat input using on_change callback (clears input safely) -----
def _on_submit():
    query = st.session_state.get("user_input", "").strip()
    if not query:
        # nothing entered — do nothing (or show warning via session_state flag)
        st.session_state.setdefault("_show_warning", True)
        return
    # compute reply and append to history
    st.session_state.setdefault("history", [])
    # perform the model call (synchronous)
    reply = decide_reply(query)
    st.session_state.history.append((query, reply))
    # clear the input by setting the session_state BEFORE rerender completes
    st.session_state["user_input"] = ""
    # clear any warning flag
    if "_show_warning" in st.session_state:
        st.session_state.pop("_show_warning", None)

# create the text input widget with an on_change callback
user_input = st.text_input("Ask Umeeda something...", key="user_input", on_change=_on_submit)

# # optional: show a Send button that simply triggers the same callback (useful for mouse users)
# if st.button("Send"):
#     _on_submit()

# # optional: display a short warning if user pressed Send with empty input (works with callback)
# if st.session_state.get("_show_warning"):
#     st.warning("Please enter a prompt before sending.")


# Display history
for u, b in st.session_state.history:
#    st.markdown(f"<div style='background:#f0f2f6;border-radius:12px;padding:8px;margin:6px 0;max-width:90%'><b>You:</b> {u}</div>", unsafe_allow_html=True)
#    st.markdown(f"<div style='background:#fff8e1;border-radius:12px;padding:8px;margin:6px 0;max-width:90%'><b>Umeeda:</b> {b}</div>", unsafe_allow_html=True)

# User message (dark text on light gray)
    st.markdown(f"""
    <div style='background:#f0f2f6;
                border-radius:12px;
                padding:8px;
                margin:6px 0;
                max-width:90%;
                color:#1C2833;'>  <!-- dark text -->
        <b style='color:#154360;'>You:</b> {u}
    </div>
    """, unsafe_allow_html=True)

# Umeeda message (dark purple text on cream background)
    st.markdown(f"""
    <div style='background:#fff8e1;
                border-radius:12px;
                padding:8px;
                margin:6px 0;
                max-width:90%;
                color:#4A235A;'>  <!-- darker text -->
        <b style='color:#6C3483;'>Umeeda:</b> {b}
    </div>
    """, unsafe_allow_html=True)
