import datetime
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from preprocess import preprocess_single_input, parse_time_to_hours
from recommendations import generate_recommendations

st.set_page_config(
    page_title="SleepIQ — Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global App Styling */
    .stApp {
        background-color: #0B1120;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.10) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(30, 41, 59, 0.3) 0px, transparent 50%);
        color: #F1F5F9;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    }

    /* Hide standard Streamlit header & hamburger */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------------- Navbar Styling ---------------- */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 28px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        margin-bottom: 32px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        font-size: 26px;
        filter: drop-shadow(0 0 10px rgba(167, 139, 250, 0.6));
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FFFFFF 30%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94A3B8;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .nav-links {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .nav-link {
        color: #94A3B8;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    .nav-link:hover {
        color: #F8FAFC;
        background: rgba(255, 255, 255, 0.06);
    }

    /* ---------------- Hero Section ---------------- */
    .hero-container {
        text-align: center;
        padding: 42px 20px 36px 20px;
        margin-bottom: 32px;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #C084FC;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #F8FAFC;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        max-width: 650px;
        margin: 0 auto 24px auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ---------------- Card Layouts ---------------- */
    .card-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 22px;
        height: 100%;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-box:hover {
        border-color: rgba(167, 139, 250, 0.3);
    }
    .card-header-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .card-icon {
        font-size: 1.3rem;
        background: rgba(99, 102, 241, 0.15);
        padding: 8px;
        border-radius: 10px;
        line-height: 1;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.2px;
    }

    /* Form Section Headings */
    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .section-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 24px;
    }

    /* ---------------- Prediction Results ---------------- */
    .result-container {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        position: relative;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .result-good {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 2px solid #10B981;
    }
    .result-average {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.2) 100%);
        border: 2px solid #F59E0B;
    }
    .result-poor {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid #EF4444;
    }

    .result-icon-large {
        font-size: 3.5rem;
        margin-bottom: 8px;
        line-height: 1;
    }
    .result-badge {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-good .result-badge { color: #34D399; text-shadow: 0 0 20px rgba(52, 211, 153, 0.4); }
    .result-average .result-badge { color: #FBBF24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.4); }
    .result-poor .result-badge { color: #F87171; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4); }

    .result-summary-text {
        font-size: 1rem;
        color: #E2E8F0;
        font-weight: 500;
        margin-top: 6px;
    }

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 14px;
        padding: 6px 14px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #CBD5E1;
    }

    /* Recommendation Tip Box */
    .tip-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 24px;
    }
    .tip-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #A78BFA;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }
    .tip-item {
        color: #E2E8F0;
        font-size: 0.94rem;
        line-height: 1.6;
        padding: 8px 12px;
        background: rgba(30, 41, 59, 0.4);
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 3px solid #8B5CF6;
    }

    /* ---------------- Analytics Box ---------------- */
    .analytics-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 4px 25px -3px rgba(0, 0, 0, 0.35);
        margin-top: 12px;
    }
    .stat-pill {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 12px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }

    /* ---------------- Minimal Footer ---------------- */
    .footer-container {
        text-align: center;
        padding: 36px 16px 16px 16px;
        margin-top: 48px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .footer-brand {
        font-size: 0.95rem;
        font-weight: 700;
        color: #E2E8F0;
        letter-spacing: -0.2px;
    }
    .footer-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Streamlit widget modifications for dark mode polish */
    div[data-baseweb="select"] {
        border-radius: 10px;
    }
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        padding: 10px 24px;
        transition: all 0.25s ease;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_bundle():
    bundle_path = os.path.join('models', 'best_model.pkl')
    if not os.path.exists(bundle_path):
        st.error("⚠️ Trained model bundle not found! Run `python train_model.py` to generate.")
        st.stop()
    return joblib.load(bundle_path)

@st.cache_data
def load_dataset():
    dataset_path = os.path.join('data', 'sleep_dataset.csv')
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

try:
    model_bundle = load_ml_bundle()
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    best_model_name = model_bundle['model_name']
    target_map = model_bundle['target_mapping']
except Exception as e:
    st.error(f"Error initializing ML system: {e}")
    st.stop()

dataset_df = load_dataset()


def reset_form_inputs():
    st.session_state['input_sleep_duration'] = 7.5
    st.session_state['input_bedtime'] = datetime.time(23, 0)
    st.session_state['input_wakeup_time'] = datetime.time(7, 0)
    st.session_state['input_caffeine'] = 'Low'
    st.session_state['input_exercise'] = 30
    st.session_state['input_screentime'] = 30
    st.session_state['input_stress'] = 3
    st.session_state['input_mood'] = 'Neutral'
    st.session_state['input_interruptions'] = 'No'
    st.session_state['has_prediction'] = False


st.markdown("""
<div class="navbar">
    <div class="nav-brand">
        <div class="brand-icon">🌙</div>
        <div>
            <div class="brand-title">SleepIQ</div>
            <div class="brand-subtitle">Sleep Quality Predictor</div>
        </div>
    </div>
    <div class="nav-links">
        <a href="#home" class="nav-link">Home</a>
        <a href="#prediction" class="nav-link">Prediction</a>
        <a href="#analytics" class="nav-link">Analytics</a>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>⚡</span> Machine Learning Powered
    </div>
    <h1 class="hero-title">Understand Your Sleep.<br>Improve Your Tomorrow.</h1>
    <p class="hero-subtitle">
        Predict your sleep quality using lifestyle, behavioral, and sleep-related factors powered by intelligent predictive modeling.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tell Us About Your Sleep</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Enter your daily sleep and lifestyle information to get your prediction.</div>', unsafe_allow_html=True)

col_sleep, col_life, col_well = st.columns(3)

with col_sleep:
    st.markdown("""
    <div class="card-header-row">
        <div class="card-icon">🛏️</div>
        <div class="card-title">Sleep Details</div>
    </div>
    """, unsafe_allow_html=True)
    
    sleep_duration = st.slider(
        "Sleep Duration (Hours)",
        min_value=3.0,
        max_value=12.0,
        value=st.session_state.get('input_sleep_duration', 7.5),
        step=0.5,
        key='input_sleep_duration',
        help="Total hours of sleep achieved overnight."
    )
    
    bedtime = st.time_input(
        "Bedtime",
        value=st.session_state.get('input_bedtime', datetime.time(23, 0)),
        key='input_bedtime',
        help="Time you typically go to sleep."
    )
    
    wakeup_time = st.time_input(
        "Wake-up Time",
        value=st.session_state.get('input_wakeup_time', datetime.time(7, 0)),
        key='input_wakeup_time',
        help="Time you wake up in the morning."
    )

with col_life:
    st.markdown("""
    <div class="card-header-row">
        <div class="card-icon">☕</div>
        <div class="card-title">Lifestyle</div>
    </div>
    """, unsafe_allow_html=True)
    
    caffeine_options = ['None', 'Low', 'Moderate', 'High']
    caffeine_intake = st.selectbox(
        "Caffeine Intake",
        options=caffeine_options,
        index=caffeine_options.index(st.session_state.get('input_caffeine', 'Low')),
        key='input_caffeine',
        help="Daily coffee, tea, energy drink intake."
    )
    
    exercise_duration = st.number_input(
        "Exercise Duration (Minutes/Day)",
        min_value=0,
        max_value=180,
        value=st.session_state.get('input_exercise', 30),
        step=5,
        key='input_exercise',
        help="Physical activity duration per day."
    )
    
    screen_time = st.number_input(
        "Screen Time Before Bed (Minutes)",
        min_value=0,
        max_value=300,
        value=st.session_state.get('input_screentime', 30),
        step=10,
        key='input_screentime',
        help="Device screen usage before sleep."
    )

with col_well:
    st.markdown("""
    <div class="card-header-row">
        <div class="card-icon">🧠</div>
        <div class="card-title">Well-being</div>
    </div>
    """, unsafe_allow_html=True)
    
    stress_level = st.slider(
        "Stress Level (0 = Relaxed, 10 = High)",
        min_value=0,
        max_value=10,
        value=st.session_state.get('input_stress', 3),
        key='input_stress',
        help="Overall daily stress and tension level."
    )
    
    mood_options = ['Happy', 'Neutral', 'Sad', 'Anxious']
    mood = st.selectbox(
        "Mood Before Sleep",
        options=mood_options,
        index=mood_options.index(st.session_state.get('input_mood', 'Neutral')),
        key='input_mood',
        help="Dominant mood or mental state before sleep."
    )
    
    sleep_interruptions = st.radio(
        "Sleep Interruptions",
        options=['No', 'Yes'],
        index=0 if st.session_state.get('input_interruptions', 'No') == 'No' else 1,
        horizontal=True,
        key='input_interruptions',
        help="Waking up during the night."
    )

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, _ = st.columns([2.5, 1.5, 6])

with btn_col1:
    predict_clicked = st.button("🔮 Predict Sleep Quality", type="primary")

with btn_col2:
    reset_clicked = st.button("🔄 Reset", on_click=reset_form_inputs)


if predict_clicked:
    raw_input_dict = {
        'sleep_duration': sleep_duration,
        'bedtime': bedtime.strftime("%H:%M"),
        'wakeup_time': wakeup_time.strftime("%H:%M"),
        'caffeine_intake': caffeine_intake,
        'exercise_duration': exercise_duration,
        'screen_time': screen_time,
        'stress_level': stress_level,
        'mood': mood,
        'sleep_interruptions': sleep_interruptions
    }

    scaled_input_df = preprocess_single_input(raw_input_dict, scaler)

    raw_prediction = model.predict(scaled_input_df)[0]
    if isinstance(raw_prediction, (int, np.integer)):
        predicted_label = target_map.get(int(raw_prediction), 'Average')
    else:
        predicted_label = str(raw_prediction)

    confidence_pct = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(scaled_input_df)[0]
            if hasattr(model, 'classes_'):
                class_idx = list(model.classes_).index(raw_prediction)
                confidence_pct = float(probabilities[class_idx]) * 100.0
            else:
                confidence_pct = float(np.max(probabilities)) * 100.0
        except Exception:
            confidence_pct = None

    rec_results = generate_recommendations(raw_input_dict, predicted_label)

    st.markdown("---")
    st.markdown('<div class="section-title">Your Sleep Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Comprehensive Machine Learning prediction based on your sleep profile.</div>', unsafe_allow_html=True)

    res_col, tip_col = st.columns([5, 7])

    with res_col:
        card_class = "result-good" if predicted_label == 'Good' else ("result-average" if predicted_label == 'Average' else "result-poor")
        icon_emoji = "🌟" if predicted_label == 'Good' else ("⚖️" if predicted_label == 'Average' else "⚠️")
        
        st.markdown(f"""
        <div class="result-container {card_class}">
            <div class="result-icon-large">{icon_emoji}</div>
            <div class="result-badge">{predicted_label.upper()}</div>
            <div class="result-summary-text">
                Sleep Quality: <strong>{predicted_label.upper()}</strong><br>
                <span style="color: #94A3B8; font-size: 0.88rem;">Based on your submitted sleep and lifestyle information.</span>
            </div>
            {"<div class='confidence-badge'>🎯 Prediction Confidence: " + f"{confidence_pct:.1f}%" + "</div>" if confidence_pct is not None else ""}
        </div>
        """, unsafe_allow_html=True)

    with tip_col:
        st.markdown("""
        <div class="tip-box">
            <div class="tip-title">💡 Sleep Improvement Tip</div>
        """, unsafe_allow_html=True)
        
        if rec_results.get('tips'):
            for tip in rec_results['tips']:
                st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-item">Maintain your consistent sleep schedule and healthy routines.</div>', unsafe_allow_html=True)

        if rec_results.get('reasons'):
            st.markdown('<div style="font-weight: 700; color: #94A3B8; font-size: 0.85rem; margin-top: 14px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Contributing Factors:</div>', unsafe_allow_html=True)
            for reason in rec_results['reasons']:
                st.markdown(f'<span style="color: #CBD5E1; font-size: 0.88rem;">• {reason}</span><br>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sleep Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Real-world statistical insights derived directly from the sleep study dataset.</div>', unsafe_allow_html=True)

if dataset_df is not None and 'sleep_quality' in dataset_df.columns:
    dist_counts = dataset_df['sleep_quality'].value_counts()
    good_count = int(dist_counts.get('Good', 0))
    avg_count = int(dist_counts.get('Average', 0))
    poor_count = int(dist_counts.get('Poor', 0))
    total_count = len(dataset_df)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-label">Total Records</div>
            <div class="stat-value">{total_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-pill" style="border-left: 3px solid #10B981;">
            <div class="stat-label" style="color: #34D399;">Good Sleep</div>
            <div class="stat-value" style="color: #34D399;">{good_count} ({good_count/total_count*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-pill" style="border-left: 3px solid #F59E0B;">
            <div class="stat-label" style="color: #FBBF24;">Average Sleep</div>
            <div class="stat-value" style="color: #FBBF24;">{avg_count} ({avg_count/total_count*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-pill" style="border-left: 3px solid #EF4444;">
            <div class="stat-label" style="color: #F87171;">Poor Sleep</div>
            <div class="stat-value" style="color: #F87171;">{poor_count} ({poor_count/total_count*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    
    categories = ['Good', 'Average', 'Poor']
    values = [good_count, avg_count, poor_count]
    colors = ['#10B981', '#F59E0B', '#EF4444']

    fig.add_trace(go.Pie(
        labels=categories,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0F172A', width=3)),
        textinfo='label+percent',
        textfont=dict(size=14, family='Plus Jakarta Sans', color='#FFFFFF'),
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label} Sleep Quality</b><br>Sample Count: %{value}<br>Share: %{percent}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="<b>Sleep Quality Distribution</b><br><span style='font-size:13px; color:#94A3B8; font-weight:normal;'>Distribution of predicted sleep quality across the dataset.</span>",
            font=dict(size=18, family='Plus Jakarta Sans', color='#F8FAFC'),
            x=0.02,
            y=0.95
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.75)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color='#CBD5E1', size=13)
        ),
        margin=dict(l=20, r=20, t=70, b=30),
        height=380
    )

    st.plotly_chart(fig, config={'displayModeBar': False})

st.markdown("""
<div class="footer-container">
    <div class="footer-brand">SleepIQ — Sleep Quality Predictor</div>
    <div class="footer-sub">Machine Learning Based Sleep Analytics</div>
</div>
""", unsafe_allow_html=True)
