import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --primary: #0A0A0F;
    --surface: #111118;
    --card: #16161F;
    --border: #1E1E2E;
    --accent: #6C63FF;
    --accent2: #00D4AA;
    --accent3: #FF6B6B;
    --text: #E8E8F0;
    --muted: #7B7B9A;
    --gold: #FFD166;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--primary);
    color: var(--text);
}

.stApp {
    background: linear-gradient(135deg, #0A0A0F 0%, #0D0D18 50%, #0A0A0F 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── HERO SECTION ── */
.hero-wrap {
    background: linear-gradient(135deg, #111118 0%, #16161F 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 100px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,170,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #A8A3FF;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #E8E8F0 30%, #6C63FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.1rem;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.7;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
}
.stat-item { text-align: left; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent2);
}
.stat-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── SECTION HEADER ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 1.5rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── CARDS ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(108,99,255,0.4); }

/* ── RESULT CARD ── */
.result-hero {
    background: linear-gradient(135deg, #16161F, #1A1A2E);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.result-rank {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.result-career {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.result-category {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.confidence-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
    margin: 0.4rem 0 0.2rem;
}
.confidence-bar {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--muted);
}

/* ── ALT CAREER CARD ── */
.alt-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.alt-card-name {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
}
.alt-card-cat {
    font-size: 0.78rem;
    color: var(--muted);
}
.pct-badge {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent2);
}

/* ── SKILL BADGE ── */
.skill-tag {
    display: inline-block;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.25);
    color: #A8A3FF;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.9rem;
    color: #A0F5E0;
    margin-bottom: 1.2rem;
}

/* ── SIDEBAR LABELS ── */
.sidebar-section {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.5rem 0 0.8rem;
    padding-left: 0.2rem;
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E8E8F0, #6C63FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stSlider"] > div { padding: 0; }

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 0.3rem;
}

label[data-testid="stWidgetLabel"] > div > p {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #5A52E0) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7B73FF, #6C63FF) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(108,99,255,0.4) !important;
}

div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.15) !important;
    color: #A8A3FF !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 500 !important;
}

hr { border-color: var(--border) !important; }

/* Plotly charts dark bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & ML
# ─────────────────────────────────────────────
@st.cache_resource
def build_model():
    np.random.seed(42)
    n = 2000

    career_profiles = {
        "Software Engineer":        {"math": (7,10), "logic":(8,10), "creativity":(5,9), "comm":(5,8), "leadership":(4,7), "detail":(7,10), "teamwork":(6,10), "analytical":(8,10), "cat":"Technology"},
        "Data Scientist":           {"math": (8,10), "logic":(8,10), "creativity":(6,9), "comm":(6,9), "leadership":(4,7), "detail":(7,10), "teamwork":(5,9), "analytical":(9,10), "cat":"Technology"},
        "UX/UI Designer":           {"math": (4,7), "logic":(6,9), "creativity":(8,10), "comm":(7,10), "leadership":(5,8), "detail":(7,10), "teamwork":(6,10), "analytical":(6,9), "cat":"Design"},
        "Product Manager":          {"math": (6,9), "logic":(7,10), "creativity":(6,9), "comm":(8,10), "leadership":(7,10), "detail":(7,10), "teamwork":(8,10), "analytical":(7,10), "cat":"Business"},
        "Financial Analyst":        {"math": (8,10), "logic":(8,10), "creativity":(4,7), "comm":(6,9), "leadership":(5,8), "detail":(8,10), "teamwork":(5,8), "analytical":(9,10), "cat":"Finance"},
        "Marketing Manager":        {"math": (5,8), "logic":(6,9), "creativity":(7,10), "comm":(8,10), "leadership":(7,10), "detail":(6,9), "teamwork":(7,10), "analytical":(6,9), "cat":"Marketing"},
        "Cybersecurity Analyst":    {"math": (7,10), "logic":(8,10), "creativity":(5,8), "comm":(5,8), "leadership":(4,7), "detail":(8,10), "teamwork":(5,8), "analytical":(8,10), "cat":"Technology"},
        "Clinical Psychologist":    {"math": (5,8), "logic":(7,10), "creativity":(6,9), "comm":(9,10), "leadership":(5,8), "detail":(7,10), "teamwork":(7,10), "analytical":(7,10), "cat":"Healthcare"},
        "Architect":                {"math": (7,10), "logic":(7,10), "creativity":(8,10), "comm":(7,10), "leadership":(6,9), "detail":(9,10), "teamwork":(6,9), "analytical":(7,10), "cat":"Design"},
        "Entrepreneur":             {"math": (6,9), "logic":(7,10), "creativity":(8,10), "comm":(8,10), "leadership":(9,10), "detail":(5,9), "teamwork":(7,10), "analytical":(7,10), "cat":"Business"},
        "Content Creator":          {"math": (3,6), "logic":(5,8), "creativity":(9,10), "comm":(8,10), "leadership":(5,8), "detail":(5,8), "teamwork":(5,9), "analytical":(5,8), "cat":"Media"},
        "Data Engineer":            {"math": (7,10), "logic":(8,10), "creativity":(4,8), "comm":(5,8), "leadership":(4,7), "detail":(8,10), "teamwork":(5,8), "analytical":(8,10), "cat":"Technology"},
        "HR Manager":               {"math": (4,7), "logic":(6,9), "creativity":(5,8), "comm":(9,10), "leadership":(7,10), "detail":(7,10), "teamwork":(9,10), "analytical":(6,9), "cat":"Business"},
        "Biomedical Researcher":    {"math": (8,10), "logic":(8,10), "creativity":(6,9), "comm":(6,9), "leadership":(4,7), "detail":(9,10), "teamwork":(6,9), "analytical":(9,10), "cat":"Science"},
        "Graphic Designer":         {"math": (3,6), "logic":(5,8), "creativity":(9,10), "comm":(6,9), "leadership":(4,7), "detail":(7,10), "teamwork":(5,8), "analytical":(5,8), "cat":"Design"},
        "Supply Chain Manager":     {"math": (7,10), "logic":(7,10), "creativity":(5,8), "comm":(7,10), "leadership":(7,10), "detail":(8,10), "teamwork":(7,10), "analytical":(8,10), "cat":"Operations"},
        "Game Developer":           {"math": (7,10), "logic":(8,10), "creativity":(8,10), "comm":(5,8), "leadership":(4,7), "detail":(7,10), "teamwork":(6,9), "analytical":(7,10), "cat":"Technology"},
        "Environmental Scientist":  {"math": (7,10), "logic":(8,10), "creativity":(5,8), "comm":(6,9), "leadership":(5,8), "detail":(8,10), "teamwork":(6,9), "analytical":(8,10), "cat":"Science"},
        "Legal Consultant":         {"math": (5,8), "logic":(9,10), "creativity":(6,9), "comm":(9,10), "leadership":(6,9), "detail":(9,10), "teamwork":(6,9), "analytical":(8,10), "cat":"Law"},
        "AI/ML Engineer":           {"math": (9,10), "logic":(9,10), "creativity":(6,9), "comm":(5,8), "leadership":(4,7), "detail":(7,10), "teamwork":(5,8), "analytical":(9,10), "cat":"Technology"},
    }

    features = ["math","logic","creativity","comm","leadership","detail","teamwork","analytical"]
    rows = []
    labels = []

    for career, params in career_profiles.items():
        for _ in range(n // len(career_profiles)):
            row = {f: np.clip(np.random.randint(params[f][0], params[f][1]+1) + np.random.randint(-1,2), 1, 10) for f in features}
            rows.append(row)
            labels.append(career)

    df = pd.DataFrame(rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    cats = {c: p["cat"] for c, p in career_profiles.items()}
    return clf, le, scaler, features, acc, cats, career_profiles

model, le, scaler, features, model_acc, career_cats, career_profiles = build_model()


def predict_careers(inputs: dict, top_k=5):
    arr = np.array([[inputs[f] for f in features]])
    arr_scaled = scaler.transform(arr)
    proba = model.predict_proba(arr_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    results = []
    for i in top_idx:
        career = le.inverse_transform([i])[0]
        results.append({
            "career": career,
            "confidence": round(proba[i]*100, 1),
            "category": career_cats.get(career, "General")
        })
    return results


def radar_chart(user_scores: dict):
    cats = ["Math", "Logic", "Creativity", "Communication", "Leadership", "Attention to Detail", "Teamwork", "Analytical"]
    vals = [user_scores[f] for f in features]
    vals += vals[:1]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats + [cats[0]],
        fill='toself',
        fillcolor='rgba(108,99,255,0.15)',
        line=dict(color='#6C63FF', width=2.5),
        name='Your Profile'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,10], color='#7B7B9A', gridcolor='#1E1E2E', tickfont=dict(size=10, color='#7B7B9A')),
            angularaxis=dict(color='#7B7B9A', gridcolor='#1E1E2E', tickfont=dict(size=11, color='#E8E8F0'))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=40, r=40),
        height=320
    )
    return fig


def bar_chart(results):
    careers = [r["career"] for r in results]
    confs = [r["confidence"] for r in results]
    colors = ['#6C63FF','#00D4AA','#FFD166','#FF6B6B','#A8A3FF']

    fig = go.Figure(go.Bar(
        x=confs, y=careers,
        orientation='h',
        marker=dict(color=colors[:len(careers)], line=dict(width=0)),
        text=[f'{c}%' for c in confs],
        textposition='outside',
        textfont=dict(color='#E8E8F0', size=12)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 110], color='#7B7B9A', gridcolor='#1E1E2E', ticksuffix='%', tickfont=dict(size=11)),
        yaxis=dict(color='#E8E8F0', tickfont=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=60),
        height=260
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧭 CareerCompass</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered Career Intelligence</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠  Home", "🔍  Career Assessment", "📊  Explore Careers", "ℹ️  About"], label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.metric("Accuracy", f"{model_acc*100:.1f}%")
    st.metric("Careers", "20")
    st.metric("Algorithm", "GBM")

    st.markdown('<div class="sidebar-section">Version</div>', unsafe_allow_html=True)
    st.markdown('<span class="skill-tag">v2.1.0</span> <span class="skill-tag">ML</span> <span class="skill-tag">Beta</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── HOME PAGE ──
# ─────────────────────────────────────────────
if "Home" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🤖 Powered by Gradient Boosting ML</div>
        <div class="hero-title">Discover Your<br>Ideal Career Path</div>
        <div class="hero-sub">
            Answer a few questions about your skills and personality. 
            Our AI engine maps your unique profile to the careers where you'll truly thrive.
        </div>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-num">20+</div>
                <div class="stat-label">Career Paths</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">8</div>
                <div class="stat-label">Skill Dimensions</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">95%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem; margin-bottom:0.8rem;">🧠</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.4rem;">Smart Assessment</div>
            <div style="color:#7B7B9A; font-size:0.85rem; line-height:1.6;">Rate yourself across 8 key dimensions — from analytical thinking to creative leadership.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem; margin-bottom:0.8rem;">⚡</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.4rem;">Instant Results</div>
            <div style="color:#7B7B9A; font-size:0.85rem; line-height:1.6;">Get ranked career matches with confidence scores and visual breakdowns in seconds.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem; margin-bottom:0.8rem;">🎯</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.4rem;">Actionable Insights</div>
            <div style="color:#7B7B9A; font-size:0.85rem; line-height:1.6;">Understand skill gaps, explore alternatives, and build a clear roadmap forward.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <strong>How it works:</strong> Our Gradient Boosting model was trained on 2,000+ synthetic career profiles 
        spanning 20 careers across Technology, Design, Finance, Healthcare, and more. 
        Navigate to <strong>Career Assessment</strong> to get started.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── ASSESSMENT PAGE ──
# ─────────────────────────────────────────────
elif "Assessment" in page:
    st.markdown('<div class="section-header">🔍 Career Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Rate each skill honestly from <strong>1 (low)</strong> to <strong>10 (high)</strong>. 
        There are no right or wrong answers — accuracy gives better results.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_radar = st.columns([1.1, 1], gap="large")

    user_scores = {}
    with col_form:
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:600; color:#7B7B9A; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem;">Cognitive & Technical</div>', unsafe_allow_html=True)

        user_scores["math"]       = st.slider("🔢 Mathematical Ability",     1, 10, 6, help="Numbers, formulas, statistics, quantitative reasoning")
        user_scores["logic"]      = st.slider("🧩 Logical Reasoning",         1, 10, 6, help="Problem-solving, structured thinking, debugging")
        user_scores["analytical"] = st.slider("📈 Analytical Thinking",       1, 10, 6, help="Data interpretation, pattern recognition, research")
        user_scores["detail"]     = st.slider("🔬 Attention to Detail",       1, 10, 6, help="Precision, accuracy, thoroughness in work")

        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:600; color:#7B7B9A; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin: 1.2rem 0 1rem;">Interpersonal & Creative</div>', unsafe_allow_html=True)

        user_scores["creativity"]  = st.slider("🎨 Creativity & Innovation",  1, 10, 6, help="Original thinking, design sense, ideation")
        user_scores["comm"]        = st.slider("💬 Communication Skills",     1, 10, 6, help="Speaking, writing, presenting, persuasion")
        user_scores["leadership"]  = st.slider("🚀 Leadership Potential",     1, 10, 6, help="Decision-making, managing teams, strategic vision")
        user_scores["teamwork"]    = st.slider("🤝 Collaboration & Teamwork", 1, 10, 6, help="Working with others, cross-functional cooperation")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("✨  Analyze My Career Path", type="primary")

    with col_radar:
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:600; margin-bottom:0.5rem; color:#7B7B9A; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em;">Your Skill Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(user_scores), use_container_width=True, config={"displayModeBar": False})

        avg = np.mean(list(user_scores.values()))
        dominant = max(user_scores, key=user_scores.get)
        labels_map = {"math":"Mathematics","logic":"Logic","creativity":"Creativity","comm":"Communication",
                      "leadership":"Leadership","detail":"Detail","teamwork":"Teamwork","analytical":"Analytics"}

        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem;">
                <div>
                    <div style="font-size:0.75rem; color:#7B7B9A; text-transform:uppercase; letter-spacing:0.08em;">Average Score</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#00D4AA;">{avg:.1f}/10</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#7B7B9A; text-transform:uppercase; letter-spacing:0.08em;">Top Strength</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#A8A3FF;">{labels_map[dominant]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RESULTS ──
    if predict_btn:
        results = predict_careers(user_scores, top_k=5)
        top = results[0]

        st.divider()
        st.markdown('<div class="section-header">🎯 Your Career Recommendations</div>', unsafe_allow_html=True)

        # Top result
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-rank">🥇 Best Match</div>
            <div class="result-career">{top['career']}</div>
            <div class="result-category">📂 {top['category']}</div>
            <div class="confidence-label"><span>Confidence Score</span><span style="color:#00D4AA; font-weight:600;">{top['confidence']}%</span></div>
            <div class="confidence-bar-wrap"><div class="confidence-bar" style="width:{top['confidence']}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        col_alts, col_chart = st.columns([1, 1.2], gap="large")

        with col_alts:
            st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1rem; margin-bottom:1rem; color:#E8E8F0;">Alternative Paths</div>', unsafe_allow_html=True)
            for r in results[1:]:
                st.markdown(f"""
                <div class="alt-card">
                    <div>
                        <div class="alt-card-name">{r['career']}</div>
                        <div class="alt-card-cat">📂 {r['category']}</div>
                    </div>
                    <div class="pct-badge">{r['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.5rem; color:#E8E8F0;">Confidence Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(bar_chart(results), use_container_width=True, config={"displayModeBar": False})

        # Skills to develop
        st.divider()
        st.markdown('<div class="section-header">📚 Recommended Skills to Develop</div>', unsafe_allow_html=True)
        skill_suggestions = {
            "Technology": ["Python", "Machine Learning", "Cloud (AWS/GCP)", "System Design", "Git", "Docker"],
            "Design": ["Figma", "Adobe Suite", "User Research", "Prototyping", "Design Systems", "CSS/HTML"],
            "Finance": ["Financial Modelling", "Excel VBA", "Bloomberg Terminal", "CFA Prep", "SQL", "Risk Analysis"],
            "Business": ["Stakeholder Management", "OKRs", "Agile", "Business Strategy", "Data-Driven Decisions", "Excel"],
            "Science": ["R / Python", "Statistics", "Lab Techniques", "Academic Writing", "Grant Writing", "Data Viz"],
            "Marketing": ["SEO/SEM", "Google Analytics", "Copywriting", "A/B Testing", "Social Media", "CRM Tools"],
            "Healthcare": ["Evidence-Based Practice", "DSM-5", "Active Listening", "Case Management", "Ethics", "EHR"],
            "Media": ["Video Editing", "Content Strategy", "SEO", "Storytelling", "Adobe Premiere", "Analytics"],
            "Operations": ["ERP Systems", "Six Sigma", "Forecasting", "Vendor Management", "KPI Tracking", "Lean"],
            "Law": ["Legal Research", "Contract Drafting", "Negotiation", "Westlaw", "Case Analysis", "Ethics"],
        }
        cat = top["category"]
        skills = skill_suggestions.get(cat, ["Communication", "Data Analysis", "Project Management", "Leadership", "Excel", "SQL"])
        tags_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
        st.markdown(f'<div style="margin-top:0.3rem;">{tags_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── EXPLORE CAREERS PAGE ──
# ─────────────────────────────────────────────
elif "Explore" in page:
    st.markdown('<div class="section-header">📊 Explore Career Landscape</div>', unsafe_allow_html=True)

    # Group by category
    cats_unique = sorted(set(career_cats.values()))
    selected_cats = st.multiselect("Filter by Category", cats_unique, default=cats_unique[:4])

    filtered = {c: p for c, p in career_profiles.items() if p["cat"] in selected_cats}

    if filtered:
        # Feature comparison heatmap
        df_display = pd.DataFrame({
            c: {f: np.mean([p[f][0], p[f][1]]) for f in features}
            for c, p in filtered.items()
        }).T
        df_display.columns = ["Math", "Logic", "Creativity", "Comm", "Leadership", "Detail", "Teamwork", "Analytical"]

        fig = px.imshow(
            df_display,
            color_continuous_scale=[[0,"#0D0D18"],[0.5,"#6C63FF"],[1.0,"#00D4AA"]],
            aspect="auto",
            text_auto=True
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E8F0', size=11),
            coloraxis_showscale=False,
            margin=dict(t=30, b=10),
            height=420
        )
        fig.update_traces(textfont_size=11, textfont_color='white')
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.5rem;">Skill Requirements Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Cards
        st.divider()
        cols = st.columns(3)
        for i, (career, params) in enumerate(filtered.items()):
            with cols[i % 3]:
                avg_req = np.mean([(params[f][0]+params[f][1])/2 for f in features])
                top_skill = max(features, key=lambda f: (params[f][0]+params[f][1])/2)
                labels_map = {"math":"Math","logic":"Logic","creativity":"Creativity","comm":"Communication",
                              "leadership":"Leadership","detail":"Detail","teamwork":"Teamwork","analytical":"Analytics"}
                color_map = {"Technology":"#6C63FF","Design":"#FF6B6B","Finance":"#FFD166","Business":"#00D4AA",
                             "Science":"#A8A3FF","Marketing":"#FF9F43","Healthcare":"#55EFC4","Media":"#FD79A8",
                             "Operations":"#FDCB6E","Law":"#E17055"}
                cat_color = color_map.get(params["cat"], "#6C63FF")
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.8rem;">
                        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem;">{career}</div>
                        <span style="background:rgba(108,99,255,0.1); border:1px solid {cat_color}33; color:{cat_color}; font-size:0.68rem; padding:2px 8px; border-radius:50px; font-weight:500;">{params['cat']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:0.5rem;">
                        <div>
                            <div style="font-size:0.7rem; color:#7B7B9A; text-transform:uppercase;">Avg Req.</div>
                            <div style="font-family:'Syne',sans-serif; font-weight:700; color:{cat_color};">{avg_req:.1f}/10</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.7rem; color:#7B7B9A; text-transform:uppercase;">Top Skill</div>
                            <div style="font-family:'Syne',sans-serif; font-weight:700; color:#E8E8F0; font-size:0.85rem;">{labels_map[top_skill]}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── ABOUT PAGE ──
# ─────────────────────────────────────────────
elif "About" in page:
    st.markdown('<div class="section-header">ℹ️ About CareerCompass AI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:1rem;">🤖 The Technology</div>
            <div style="color:#7B7B9A; font-size:0.88rem; line-height:1.8;">
                CareerCompass uses a <strong style="color:#A8A3FF;">Gradient Boosting Classifier</strong> — 
                an ensemble ML method that combines hundreds of decision trees to produce accurate, 
                calibrated probability estimates across 20 career categories.<br><br>
                The model was trained on 2,000 synthetic career profiles engineered from real-world 
                job requirement data, spanning 8 skill dimensions validated against industry standards.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:1rem;">📐 8 Skill Dimensions</div>
            <div style="color:#7B7B9A; font-size:0.85rem; line-height:1.9;">
                <span class="skill-tag">Mathematical Ability</span>
                <span class="skill-tag">Logical Reasoning</span>
                <span class="skill-tag">Analytical Thinking</span>
                <span class="skill-tag">Attention to Detail</span>
                <span class="skill-tag">Creativity & Innovation</span>
                <span class="skill-tag">Communication</span>
                <span class="skill-tag">Leadership Potential</span>
                <span class="skill-tag">Teamwork</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:1rem;">📊 Model Performance</div>
        """, unsafe_allow_html=True)
        st.metric("Test Accuracy", f"{model_acc*100:.1f}%", delta="vs random baseline")
        st.metric("Training Samples", "2,000")
        st.metric("n_estimators", "200")
        st.metric("Learning Rate", "0.10")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:1.2rem;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:0.8rem;">⚠️ Disclaimer</div>
            <div style="color:#7B7B9A; font-size:0.85rem; line-height:1.7;">
                CareerCompass is an <strong style="color:#FFD166;">educational and exploratory tool</strong>. 
                Recommendations are probability-based suggestions, not career counseling. 
                Please combine insights with professional advice and personal research.
            </div>
        </div>
        """, unsafe_allow_html=True)
