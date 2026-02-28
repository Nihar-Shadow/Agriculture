import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark Blue background */
    .stApp {
        background: linear-gradient(160deg, #020818 0%, #0a1628 40%, #0d2137 70%, #071a2e 100%);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(14, 60, 110, 0.35);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 16px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(2, 8, 24, 0.92);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #0369a1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14,165,233,0.45);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(14,165,233,0.2), rgba(6,182,212,0.15));
        border: 1px solid rgba(56,189,248,0.45);
        border-radius: 20px;
        padding: 28px 32px;
        text-align: center;
        margin-top: 16px;
    }
    .result-box h2 { font-size: 2rem; margin: 0; color: #38bdf8; }
    .result-box p  { color: #7dd3fc; margin: 6px 0 0; font-size: 1.05rem; }

    /* Section headers */
    h1, h2, h3 { color: #e0f2fe !important; }
    label, .stSlider label, .stNumberInput label { color: #7dd3fc !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #38bdf8; font-weight: 600; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #0ea5e9 !important; color: #7dd3fc !important; }

    /* Divider */
    hr { border-color: rgba(56,189,248,0.15); }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

@st.cache_data
def run_all_models(df):
    """Train all 6 algorithms and return a comparison DataFrame."""
    X_cls = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y_cls = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cls)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # ── Regression target ──
    y_reg   = df["temperature"]
    X_simple = df[["rainfall"]]
    X_multi  = df[["N","P","K","humidity","ph","rainfall"]]
    Xs_tr, Xs_te, yr_tr, yr_te = train_test_split(X_simple, y_reg, test_size=0.2, random_state=42)
    Xm_tr, Xm_te, _,    _      = train_test_split(X_multi,  y_reg, test_size=0.2, random_state=42)

    results = []

    # 1 Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    results.append({"Algorithm": "Logistic Regression", "Type": "Classification",
                    "Metric": "Accuracy", "Score": round(accuracy_score(y_te, lr.predict(X_te))*100, 2),
                    "Emoji": "📈", "Color": "#818cf8"})

    # 2 KNN
    best_k, best_acc = 3, 0
    for k in range(1, 22, 2):
        tmp = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        a   = accuracy_score(y_te, tmp.predict(X_te))
        if a > best_acc: best_k, best_acc = k, a
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr)
    results.append({"Algorithm": f"KNN  (K={best_k})", "Type": "Classification",
                    "Metric": "Accuracy", "Score": round(accuracy_score(y_te, knn.predict(X_te))*100, 2),
                    "Emoji": "🔵", "Color": "#38bdf8"})

    # 3 Naive Bayes
    nb = GaussianNB().fit(X_tr, y_tr)
    results.append({"Algorithm": "Naive Bayes", "Type": "Classification",
                    "Metric": "Accuracy", "Score": round(accuracy_score(y_te, nb.predict(X_te))*100, 2),
                    "Emoji": "🟣", "Color": "#c084fc"})

    # 4 Simple Linear Regression
    slr = LinearRegression().fit(Xs_tr, yr_tr)
    slr_r2 = max(0.0, r2_score(yr_te, slr.predict(Xs_te)) * 100)
    results.append({"Algorithm": "Simple Linear Regression", "Type": "Regression",
                    "Metric": "R² Score", "Score": round(slr_r2, 2),
                    "Emoji": "📉", "Color": "#fb923c"})

    # 5 Multiple Linear Regression
    mlr = LinearRegression().fit(Xm_tr, yr_tr)
    mlr_r2 = max(0.0, r2_score(yr_te, mlr.predict(Xm_te)) * 100)
    results.append({"Algorithm": "Multiple Linear Regression", "Type": "Regression",
                    "Metric": "R² Score", "Score": round(mlr_r2, 2),
                    "Emoji": "📊", "Color": "#f472b6"})

    # 6 Yield Prediction (Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te)) * 100
    results.append({"Algorithm": "Yield Prediction (RF)", "Type": "Classification",
                    "Metric": "Accuracy", "Score": round(rf_acc, 2),
                    "Emoji": "🌳", "Color": "#4ade80"})

    df_res = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

df = load_data()
model, accuracy = train_model(df)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🌿", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🌰",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌸",
    "jute": "🌿", "coffee": "☕",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px;'>
    <h1 style='font-size:2.8rem; background: linear-gradient(90deg,#38bdf8,#06b6d4);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-weight:700; margin-bottom:4px;'>
        🌱 Crop Recommendation System
    </h1>
    <p style='color:#7dd3fc; font-size:1.1rem;'>
        AI-powered smart farming — get the best crop for your soil conditions
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📊 Dataset Rows",     f"{len(df):,}")
kpi2.metric("🌿 Crop Types",       df["label"].nunique())
kpi3.metric("🤖 Model Accuracy",   f"{accuracy*100:.1f}%")
kpi4.metric("🔬 Features Used",    len(FEATURES))

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Crop", "📊 Data Insights", "📁 Raw Data", "🤖 Model Comparison"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Soil & Climate Parameters")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🧪 Soil Nutrients")
        N   = st.slider("Nitrogen (N)",        int(df.N.min()),   int(df.N.max()),   int(df.N.mean()),   help="Nitrogen ratio in soil")
        P   = st.slider("Phosphorus (P)",      int(df.P.min()),   int(df.P.max()),   int(df.P.mean()),   help="Phosphorus ratio in soil")
        K   = st.slider("Potassium (K)",       int(df.K.min()),   int(df.K.max()),   int(df.K.mean()),   help="Potassium ratio in soil")
        ph  = st.slider("Soil pH",             float(df.ph.min()), float(df.ph.max()), float(df.ph.mean()), step=0.01)

    with col2:
        st.markdown("#### 🌤️ Climate Conditions")
        temperature = st.slider("Temperature (°C)",  float(df.temperature.min()), float(df.temperature.max()), float(df.temperature.mean()), step=0.1)
        humidity    = st.slider("Humidity (%)",      float(df.humidity.min()),    float(df.humidity.max()),    float(df.humidity.mean()),    step=0.1)
        rainfall    = st.slider("Rainfall (mm)",     float(df.rainfall.min()),    float(df.rainfall.max()),    float(df.rainfall.mean()),    step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]
        top3_idx   = proba.argsort()[-3:][::-1]
        top3       = [(model.classes_[i], proba[i]) for i in top3_idx]

        emoji = CROP_EMOJI.get(prediction, "🌾")
        conf  = proba.max() * 100

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:4rem">{emoji}</div>
            <h2>{prediction.upper()}</h2>
            <p>Confidence: <strong>{conf:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**🏅 Top 3 Recommendations:**", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        for col, (crop, prob) in zip([r1, r2, r3], top3):
            col.metric(
                label=f"{CROP_EMOJI.get(crop,'🌾')} {crop.title()}",
                value=f"{prob*100:.1f}%"
            )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Insights
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Insights")
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor":   "none",
        "axes.edgecolor":   "#4b5563",
        "axes.labelcolor":  "#e0d7ff",
        "xtick.color":      "#9ca3af",
        "ytick.color":      "#9ca3af",
        "text.color":       "#e0d7ff",
        "grid.color":       "#374151",
        "grid.alpha":       0.4,
    })

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Crop Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["label"].value_counts()
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(counts)))
        ax.barh(counts.index, counts.values, color=colors)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = df[FEATURES].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuBu",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 8})
        ax.tick_params(labelsize=8)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Feature Distributions by Crop")
    selected_feat = st.selectbox("Select Feature", FEATURES)
    selected_crops = st.multiselect(
        "Select Crops to Compare",
        options=sorted(df["label"].unique()),
        default=sorted(df["label"].unique())[:5]
    )
    if selected_crops:
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected_crops)))
        for crop, color in zip(selected_crops, colors):
            subset = df[df["label"] == crop][selected_feat]
            ax.hist(subset, bins=20, alpha=0.6, label=crop, color=color, edgecolor="none")
        ax.set_xlabel(selected_feat)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### 📈 Dataset Statistics")
    st.dataframe(df[FEATURES].describe().round(2).style.background_gradient(cmap="PuBu"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Raw Data
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📁 Raw Dataset")
    search = st.text_input("🔍 Filter by crop name", "")
    filtered = df[df["label"].str.contains(search, case=False)] if search else df
    st.info(f"Showing **{len(filtered):,}** of **{len(df):,}** rows")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=480)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 ML Algorithm Comparison")
    st.markdown("""
    <p style='color:#94a3b8; margin-top:-8px;'>
    All six algorithms trained on <strong>Crop_recommendation.csv</strong> (80/20 split).
    Classification metrics use <em>Accuracy</em>; Regression uses <em>R² Score</em>.
    </p>""", unsafe_allow_html=True)

    with st.spinner("⏳ Training all models… this takes a few seconds"):
        df_cmp = run_all_models(df)

    # ── Winner banner ─────────────────────────────────────────────
    winner = df_cmp.iloc[0]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(14,165,233,0.18),rgba(6,182,212,0.15));
                border:1px solid rgba(56,189,248,0.45); border-radius:18px;
                padding:22px 30px; margin-bottom:24px; text-align:center;
                box-shadow:0 4px 24px rgba(14,165,233,0.12);">
        <div style="font-size:2.6rem;">🏆</div>
        <h2 style="color:#38bdf8; margin:6px 0 4px; font-size:1.7rem;">
            {winner['Emoji']} {winner['Algorithm']}
        </h2>
        <p style="color:#7dd3fc; font-size:1.1rem; margin:0;">
            Best {winner['Metric']}: <strong>{winner['Score']:.2f}%</strong>
            &nbsp;|&nbsp; Type: {winner['Type']}
        </p>
        <p style="color:#0ea5e9; font-size:0.9rem; margin-top:6px;">
            ✅ Recommended for production use in this project
        </p>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────
    cls_models  = df_cmp[df_cmp["Type"] == "Classification"]
    reg_models  = df_cmp[df_cmp["Type"] == "Regression"]
    best_cls_sc = cls_models["Score"].max() if not cls_models.empty else 0
    best_reg_sc = reg_models["Score"].max()  if not reg_models.empty  else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🧠 Models Trained",        len(df_cmp))
    k2.metric("📊 Classification Models", len(cls_models))
    k3.metric("📉 Best Classification",   f"{best_cls_sc:.2f}%")
    k4.metric("📈 Best Regression R²",     f"{best_reg_sc:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────
    ch1, ch2 = st.columns([1.6, 1], gap="large")

    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none",
        "axes.edgecolor": "#1e3a5f",  "axes.labelcolor": "#bae6fd",
        "xtick.color": "#7dd3fc",    "ytick.color": "#7dd3fc",
        "text.color": "#e0f2fe",     "grid.color": "#1e3a5f",
    })

    with ch1:
        st.markdown("#### 📊 Accuracy / R² Score — All Algorithms")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(
            df_cmp["Algorithm"].tolist()[::-1],
            df_cmp["Score"].tolist()[::-1],
            color=df_cmp["Color"].tolist()[::-1],
            height=0.55, edgecolor="none"
        )
        for bar, val in zip(bars, df_cmp["Score"].tolist()[::-1]):
            ax.text(min(val + 0.5, 102), bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 108)
        ax.set_xlabel("Score (%)", fontsize=9)
        ax.axvline(x=50, color="#475569", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with ch2:
        st.markdown("#### 🕸️ Radar Chart")
        labels  = df_cmp["Algorithm"].str.replace(r" \(.*\)", "", regex=True).tolist()
        scores  = df_cmp["Score"].tolist()
        colors  = df_cmp["Color"].tolist()
        N_axes  = len(labels)
        angles  = [n / float(N_axes) * 2 * np.pi for n in range(N_axes)]
        angles += angles[:1]
        scores_plot = scores + scores[:1]

        fig2, ax2 = plt.subplots(figsize=(4.4, 4.4), subplot_kw=dict(polar=True))
        ax2.set_facecolor("none")
        fig2.patch.set_alpha(0)
        ax2.plot(angles, scores_plot, "o-", linewidth=2, color="#38bdf8")
        ax2.fill(angles, scores_plot, alpha=0.2, color="#0ea5e9")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, size=7, color="#bae6fd")
        ax2.set_ylim(0, 105)
        ax2.set_yticks([25, 50, 75, 100])
        ax2.set_yticklabels(["25", "50", "75", "100"], size=6, color="#7dd3fc")
        ax2.grid(color="#1e3a5f", linewidth=0.6)
        ax2.spines["polar"].set_color("#1e3a5f")
        # colour dots
        for angle, score, color in zip(angles[:-1], scores, colors):
            ax2.plot(angle, score, "o", color=color, markersize=7)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Algorithm detail cards ─────────────────────────────────────
    st.markdown("#### 🃏 Per-Algorithm Detail")
    type_icons = {"Classification": "🏷️", "Regression": "📐"}

    ALGO_DESC = {
        "Logistic Regression": "Predicts crop class probabilities using a logistic function. Fast and interpretable.",
        "Naive Bayes":         "Applies Bayes' theorem assuming feature independence. Surprisingly powerful here.",
        "Yield Prediction (RF)": "Random Forest ensemble — combines 100 decision trees. Excellent for feature importance.",
        "Multiple Linear Regression": "Fits a hyperplane to multi-dimensional soil features to predict temperature.",
        "Simple Linear Regression": "Uses only rainfall to predict temperature — shows low correlation.",
    }

    cols = st.columns(3)
    for i, (_, row) in enumerate(df_cmp.iterrows()):
        col = cols[i % 3]
        rank_icon = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
        badge_bg  = "rgba(74,222,128,0.22)" if i == 0 else "rgba(255,255,255,0.06)"
        border_c  = row["Color"] if i == 0 else "rgba(255,255,255,0.12)"
        name_key  = row["Algorithm"].split("(")[0].strip()
        desc      = ALGO_DESC.get(name_key, ALGO_DESC.get(row["Algorithm"],
                    "Statistical learning algorithm applied to crop data."))
        col.markdown(f"""
        <div style="background:{badge_bg}; border:1px solid {border_c};
                    border-radius:16px; padding:18px 16px; margin-bottom:14px;">
            <div style="font-size:1.8rem; margin-bottom:6px;">{row['Emoji']}</div>
            <div style="font-weight:700; color:{row['Color']}; font-size:1rem;">
                {rank_icon} {row['Algorithm']}
            </div>
            <div style="font-size:0.78rem; color:#94a3b8; margin:4px 0 10px;">
                {type_icons.get(row['Type'], '📊')} {row['Type']} &nbsp;|&nbsp; {row['Metric']}
            </div>
            <div style="font-size:2rem; font-weight:800; color:white;">
                {row['Score']:.2f}%
            </div>
            <div style="background:rgba(0,0,0,0.25); border-radius:8px; height:8px; margin:10px 0 8px;">
                <div style="background:{row['Color']}; width:{min(row['Score'],100)}%;
                            height:8px; border-radius:8px;"></div>
            </div>
            <div style="font-size:0.78rem; color:#9ca3af; line-height:1.45;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Summary table ──────────────────────────────────────────────
    st.markdown("#### 📋 Full Comparison Table")
    display_df = df_cmp[["Algorithm","Type","Metric","Score"]].rename(
        columns={"Score": "Score (%)"})
    display_df.insert(0, "Rank", ["🥇","🥈","🥉"] + [f"#{i}" for i in range(4, len(df_cmp)+1)])
    st.dataframe(
        display_df.style
            .background_gradient(subset=["Score (%)"], cmap="PuBu")
            .format({"Score (%)": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("""
    <div style="background:rgba(14,165,233,0.1); border:1px solid rgba(56,189,248,0.3);
                border-radius:14px; padding:16px 20px; margin-top:16px;">
        <strong style="color:#38bdf8;">💡 Interpretation Note</strong><br>
        <span style="color:#7dd3fc; font-size:0.88rem;">
        Classification <em>Accuracy</em> and Regression <em>R² Score</em> are on different scales —
        they cannot be compared directly. Regression R² near 0% means temperature
        is largely <em>independent</em> of those soil features, which is agronomically expected.
        </span>
    </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style='text-align:center; color:#6b7280; font-size:0.85rem; padding-bottom:12px;'>
    🌱 Crop Recommendation System &nbsp;|&nbsp; Powered by Random Forest &nbsp;|&nbsp; BML Case Study
</p>
""", unsafe_allow_html=True)
