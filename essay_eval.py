# essay_eval.py
import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Config
# -----------------------
MODEL_ID_DEFAULT = "Kevintu/Engessay_grading_ML"
TRAITS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

st.set_page_config(page_title="Essay Trait Evaluator", page_icon="📝", layout="wide")

# -----------------------
# Utilities
# -----------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_id: str):
    """
    Cached model/tokenizer loader. First run needs internet to download the model.
    Subsequent runs use the local cache.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval().to(DEVICE)
    return tokenizer, model

def encode_with_sliding_window(text, tokenizer, max_len=512, stride=128):
    """
    Create overlapping token windows for very long essays; returns a list of encodings (dicts).
    """
    # Fast short-circuit if it easily fits
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_len)
    if enc["input_ids"].shape[-1] < max_len:
        return [enc]

    # Token-level windowing
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.prepare_for_model(
            chunk_tokens, return_tensors="pt", truncation=True, max_length=max_len
        )
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(end - stride, 0)
    return chunks

def map_logits_to_scores(logits: np.ndarray, clamp_min=1.0, clamp_max=10.0, round_to_half=True):
    """
    Provisional mapping from logits to 1–10.
    NOTE: Replace with the official mapping if the model card provides one
          or after you calibrate to human-scored anchors.
    """
    scores = 2.25 * logits - 1.25  # <— provisional mapping
    scores = np.clip(scores, clamp_min, clamp_max)
    if round_to_half:
        scores = np.round(scores * 2) / 2.0
    return scores

def predict_scores(text: str,
                   tokenizer,
                   model,
                   max_len: int = 512,
                   use_sliding: bool = False,
                   stride: int = 128):
    """
    Tokenize (with optional sliding windows), run model, average logits across windows,
    and map to 1–10 scores.
    """
    if not text.strip():
        raise ValueError("Empty essay text.")

    if use_sliding:
        encodings = encode_with_sliding_window(text, tokenizer, max_len=max_len, stride=stride)
    else:
        encodings = [tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)]

    all_logits = []
    for enc in encodings:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.squeeze(0).detach().cpu().numpy()
        if logits.shape[-1] != len(TRAITS):
            raise ValueError(f"Model returned {logits.shape[-1]} dims, expected {len(TRAITS)}.")
        all_logits.append(logits)

    avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    scores = map_logits_to_scores(avg_logits)
    return avg_logits, scores

# -----------------------
# UI
# -----------------------
st.title("📝 Essay Trait Evaluator")
st.caption(
    "Estimates analytic trait scores per essay using a Hugging Face model. "
    "Results are provisional; instructors make final decisions."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_id = st.text_input("Model ID", value=MODEL_ID_DEFAULT, help="Any compatible HF model repo ID.")
    max_len = st.slider("Max token length", min_value=128, max_value=2048, value=512, step=64,
                        help="Tokenizer truncation length (per window).")
    use_sliding = st.checkbox("Use sliding window for long essays", value=False,
                              help="Enables overlapping windows to reduce front-loaded bias.")
    stride = st.slider("Sliding window stride", min_value=32, max_value=1024, value=128, step=32,
                       help="Token overlap between windows (only used if sliding is enabled).")
    show_raw = st.checkbox("Show raw logits", value=False)
    st.markdown("---")
    st.write(f"**Device:** `{DEVICE}`")

# Load model/tokenizer (cached)
try:
    tokenizer, model = load_model_and_tokenizer(model_id)
    st.success(f"Loaded `{model_id}` on `{DEVICE}`")
except Exception as e:
    st.error(
        "Model load failed. For the first run, ensure this machine has internet access to download the model. "
        f"\n\n**Error:** {e}"
    )
    st.stop()

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Paste Essay")
    essay_text = st.text_area(
        "Essay text",
        value="",
        height=260,
        placeholder="Paste your essay here…"
    )
    uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
    if uploaded is not None:
        try:
            essay_text = uploaded.read().decode("utf-8", errors="ignore")
            st.info("Loaded text from file.")
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    run = st.button("Evaluate Essay", type="primary")

with col_right:
    st.subheader("Notes")
    st.markdown(
        "- Uses a **provisional** logits→score mapping (1–10). Replace it after calibration to your rubric.\n"
        "- For very long essays, enable **Sliding Window** to average predictions across chunks.\n"
        "- Scores are rounded to **0.5** after clamping to [1, 10]."
    )

st.markdown("---")

# -----------------------
# Inference & Results
# -----------------------
if run:
    if not essay_text.strip():
        st.warning("Please enter or upload an essay.")
        st.stop()

    with st.spinner("Evaluating…"):
        try:
            raw_logits, scores = predict_scores(
                essay_text, tokenizer, model, max_len=max_len,
                use_sliding=use_sliding, stride=stride
            )
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            st.stop()

    # Assemble results
    df = pd.DataFrame({
        "trait": TRAITS,
        "score (1–10)": scores
    })

    st.subheader("📊 Analytic Trait Scores")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Simple bar chart
    chart_data = df.set_index("trait")
    st.bar_chart(chart_data)

    # Optional raw logits
    if show_raw:
        st.code(np.array2string(raw_logits, precision=4, separator=", "), language="text")

    # Download CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download scores as CSV",
        data=csv_bytes,
        file_name=f"essay_trait_scores_{timestamp}.csv",
        mime="text/csv"
    )

    # Provenance + reminder
    with st.expander("Method & caveats"):
        st.markdown(
            "- Model: `{}`.\n"
            "- Mapping: `scores = clip(2.25 * logits - 1.25, 1, 10)` then rounded to 0.5.\n"
            "- Consider calibrating to your rubric with human-scored anchors.".format(model_id)
        )
