"""
Multilingual Agricultural Advisory System
==========================================
Run with: streamlit run agri_advisor.py

Requirements:
    pip install streamlit anthropic deep-translator
"""

import streamlit as st
import anthropic
import json
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroAdvisor – Multilingual Farm Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

/* Earthy palette */
:root {
    --soil: #4A3728;
    --straw: #D4A853;
    --leaf: #3D7A3B;
    --sky: #5B9BD5;
    --cream: #FAF6EF;
    --rust: #C0522A;
}

.main { background-color: var(--cream); }

.stApp { background: linear-gradient(160deg, #FAF6EF 0%, #EEF7ED 100%); }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #3D7A3B 0%, #2C5F2E 60%, #4A3728 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🌾";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.2;
}
.hero-banner h1 {
    margin: 0;
    font-size: 2.2rem;
    color: white !important;
    text-shadow: 1px 2px 6px rgba(0,0,0,0.3);
}
.hero-banner p {
    margin: 0.3rem 0 0;
    opacity: 0.85;
    font-size: 1rem;
}

/* Category cards */
.cat-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-left: 5px solid var(--leaf);
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    cursor: pointer;
    margin-bottom: 0.6rem;
    transition: box-shadow 0.2s;
}
.cat-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.13); }

/* Advisory response card */
.advisory-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-top: 5px solid #3D7A3B;
    margin-top: 1rem;
}

/* Translation box */
.translation-box {
    background: #EEF7ED;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #c3e0c2;
    margin-top: 0.8rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2C5F2E 0%, #1a3d1b 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #d8f5d8 !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #a8d5a8 !important;
    font-family: 'Playfair Display', serif !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3D7A3B, #2C5F2E);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(61,122,59,0.4);
}

.stTextArea textarea, .stTextInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #d0e8cf !important;
}

/* Info chips */
.chip {
    display: inline-block;
    background: #e8f5e7;
    color: #2C5F2E;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE MAP
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "Portuguese (Português)": "pt",
    "Swahili (Kiswahili)": "sw",
    "Bengali (বাংলা)": "bn",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Tamil (தமிழ்)": "ta",
    "Urdu (اردو)": "ur",
    "Arabic (العربية)": "ar",
    "Mandarin (普通话)": "zh-CN",
    "Indonesian (Bahasa)": "id",
    "Hausa": "ha",
    "Amharic (አማርኛ)": "am",
}

ADVISORY_CATEGORIES = {
    "🌱 Crop Selection & Planning": "crop selection, planting schedules, crop rotation, and variety recommendations",
    "🐛 Pest & Disease Management": "pest identification, disease control, integrated pest management (IPM), and safe pesticide use",
    "💧 Irrigation & Water Management": "irrigation scheduling, water conservation, drought-resistant practices, and soil moisture",
    "🌿 Soil Health & Fertilization": "soil testing, fertilizer recommendations, composting, and nutrient management",
    "☀️ Weather & Climate Adaptation": "seasonal planning, climate-smart agriculture, frost protection, and weather risk",
    "🐄 Livestock & Poultry": "animal nutrition, disease prevention, breeding, and herd/flock management",
    "📦 Post-Harvest & Storage": "harvesting timing, storage techniques, reducing post-harvest losses, and food safety",
    "💰 Market & Finance": "crop pricing, government schemes, farm loans, insurance, and market linkage",
    "🌳 Agroforestry & Sustainability": "tree integration, organic farming, biodiversity, and sustainable practices",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_client():
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def translate_text(text: str, src: str, dest: str) -> str:
    if src == dest or dest == "en":
        return text
    try:
        translator = GoogleTranslator(source=src, target=dest)
        # chunk large text to avoid limits
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        return " ".join(translator.translate(c) for c in chunks)
    except Exception as e:
        return f"[Translation unavailable: {e}]\n\n{text}"


def build_system_prompt(category_desc: str, region: str, season: str) -> str:
    return f"""You are AgroAdvisor, an expert multilingual agricultural extension officer with deep knowledge of global farming practices.

Context:
- Advisory category: {category_desc}
- Farmer's region/location: {region if region else "not specified"}
- Current season: {season if season else "not specified"}

Instructions:
1. Provide clear, practical, and actionable agricultural advice.
2. Use simple language understandable by small-scale farmers.
3. Structure your answer with: a brief diagnosis/overview, step-by-step recommendations, and important precautions.
4. Where relevant, mention low-cost or locally available alternatives.
5. If the query is outside agricultural topics, politely redirect.
6. Always reply in English (translation will be handled separately).
7. Keep responses focused and under 400 words unless detail is critical."""


def get_advisory(client, query: str, category: str, region: str, season: str) -> str:
    cat_desc = ADVISORY_CATEGORIES.get(category, "general agriculture")
    messages = st.session_state.chat_history + [{"role": "user", "content": query}]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=build_system_prompt(cat_desc, region, season),
        messages=messages,
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("### 🌍 Language")
    input_lang_name = st.selectbox("Your Language (Input)", list(LANGUAGES.keys()), index=0)
    output_lang_name = st.selectbox("Response Language (Output)", list(LANGUAGES.keys()), index=0)

    st.markdown("### 📍 Farm Context")
    region = st.text_input("Region / Country", placeholder="e.g. Punjab, India")
    season = st.selectbox("Current Season", ["", "Spring", "Summer", "Monsoon", "Autumn", "Winter", "Dry Season", "Wet Season"])

    st.markdown("### 📂 Advisory Category")
    category = st.selectbox("Select Topic", list(ADVISORY_CATEGORIES.keys()))

    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.session_state.last_query = None
        st.rerun()

    st.markdown(
        """
<div style='margin-top:2rem; font-size:0.78rem; color:#a8d5a8; line-height:1.6'>
<b>AgroAdvisor v1.0</b><br>
Powered by Claude AI<br>
Supports 15+ languages<br>
Available 24/7 for farmers
</div>
""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
    <h1>🌾 AgroAdvisor</h1>
    <p>Multilingual Agricultural Advisory System · Ask in your language, get expert guidance instantly</p>
</div>
""",
    unsafe_allow_html=True,
)

# Quick-question chips
st.markdown("**Quick Questions:**")
quick_cols = st.columns(4)
quick_questions = {
    "🐛 Pest on my wheat": "My wheat crop has small black insects on the leaves and the tips are turning yellow. What pest is this and how do I treat it?",
    "💧 Irrigation schedule": "How often should I irrigate my tomato crop during the dry season?",
    "🌱 Best monsoon crops": "Which crops are best to plant at the start of monsoon season in a tropical region?",
    "🌿 Soil improvement": "My soil is sandy and drains too fast. How can I improve its water retention for vegetables?",
}
for i, (label, question) in enumerate(quick_questions.items()):
    with quick_cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state.prefill_query = question

# Main query area
prefill = st.session_state.pop("prefill_query", "")
input_lang_code = LANGUAGES[input_lang_name]
output_lang_code = LANGUAGES[output_lang_name]

query = st.text_area(
    f"Your Question ({input_lang_name})",
    value=prefill,
    height=120,
    placeholder="Describe your farming problem or question here… / अपनी समस्या यहाँ लिखें… / Écrivez votre question ici…",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🌿 Get Advice", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────────────────────────────────────────
if submit and query.strip():
    client = get_client()
    if not client:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
    else:
        with st.spinner("🌱 Consulting our agricultural experts…"):
            # Translate input to English if needed
            if input_lang_code != "en":
                english_query = translate_text(query, input_lang_code, "en")
            else:
                english_query = query

            # Get advisory
            try:
                advisory_en = get_advisory(client, english_query, category, region, season)

                # Update history
                st.session_state.chat_history.append({"role": "user", "content": english_query})
                st.session_state.chat_history.append({"role": "assistant", "content": advisory_en})

                # Translate output
                if output_lang_code != "en":
                    advisory_translated = translate_text(advisory_en, "en", output_lang_code)
                else:
                    advisory_translated = advisory_en

                st.session_state.last_query = query
                st.session_state.last_response = {
                    "en": advisory_en,
                    "translated": advisory_translated,
                    "output_lang": output_lang_name,
                    "output_code": output_lang_code,
                    "category": category,
                }
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESPONSE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.last_response:
    resp = st.session_state.last_response
    st.markdown(
        f"""
<div class="advisory-card">
    <div style="margin-bottom:0.8rem">
        <span class="chip">{resp['category']}</span>
        <span class="chip">📍 {region or 'Any Region'}</span>
        <span class="chip">🗓️ {season or 'Any Season'}</span>
    </div>
    <h3 style="color:#2C5F2E; margin-top:0">Agricultural Advisory</h3>
""",
        unsafe_allow_html=True,
    )

    if resp["output_code"] != "en":
        tab1, tab2 = st.tabs([f"🌍 {resp['output_lang']}", "🇬🇧 English"])
        with tab1:
            st.markdown(
                f'<div class="translation-box">{resp["translated"]}</div>',
                unsafe_allow_html=True,
            )
        with tab2:
            st.markdown(resp["en"])
    else:
        st.markdown(resp["en"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Download button
    dl_content = f"AgroAdvisor Report\n{'='*40}\nQuery: {st.session_state.last_query}\n\nAdvisory ({resp['output_lang']}):\n{resp['translated']}\n\nAdvisory (English):\n{resp['en']}"
    st.download_button(
        "📄 Download Advisory Report",
        data=dl_content,
        file_name="agro_advisory.txt",
        mime="text/plain",
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if len(st.session_state.chat_history) > 2:
    with st.expander(f"💬 Conversation History ({len(st.session_state.chat_history)//2} exchanges)", expanded=False):
        for i in range(0, len(st.session_state.chat_history) - 2, 2):
            user_msg = st.session_state.chat_history[i]["content"]
            asst_msg = st.session_state.chat_history[i + 1]["content"]
            st.markdown(f"**🧑‍🌾 You:** {user_msg}")
            st.markdown(f"**🌿 Advisor:** {asst_msg[:300]}{'…' if len(asst_msg) > 300 else ''}")
            st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GUIDE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.last_response:
    st.markdown("---")
    st.markdown("### 🚀 How to Use AgroAdvisor")
    cols = st.columns(3)
    features = [
        ("🌍 Multilingual", "Ask questions in 15+ languages. Responses auto-translated to your preferred language."),
        ("🤖 AI-Powered", "Claude AI provides expert-level agricultural guidance tailored to your context."),
        ("📋 Multi-topic", "Covers crops, pests, irrigation, soil, livestock, markets, and more."),
        ("📍 Context-Aware", "Set your region and season for highly relevant local recommendations."),
        ("💬 Multi-turn Chat", "Follow up with more questions — the advisor remembers your conversation."),
        ("📄 Downloadable", "Save advisory reports as text files for offline reference in the field."),
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
<div class="cat-card">
    <b>{title}</b><br>
    <span style="font-size:0.88rem; color:#555">{desc}</span>
</div>
""",
                unsafe_allow_html=True,
            )
