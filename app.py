import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
model = joblib.load('xgb_pipeline.pkl')

st.set_page_config(page_title="Fraud AI", layout="centered")

# ===== Animated CSS =====
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #0b1220, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5ff, #7c3aed, #00f5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 🔥 Labels (names of fields) */
label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

/* Inputs spacing */
div[data-baseweb="input"] {
    margin-bottom: 15px;
}

/* Selectbox text */
div[data-baseweb="select"] {
    color: white !important;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(0,255,255,0.1);
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00f5ff, #7c3aed);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    padding: 12px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px #00f5ff;
}

</style>
""", unsafe_allow_html=True)

# ===== Title =====
st.markdown("<div class='title'>💳 Fraud Detection AI</div>", unsafe_allow_html=True)
st.write("")

# ===== Inputs Card =====
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    type_trans = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    amount = st.number_input("Amount")

    oldbalanceOrg = st.number_input("Sender Old Balance")

with col2:
    newbalanceOrig = st.number_input("Sender New Balance")
    oldbalanceDest = st.number_input("Receiver Old Balance")
    newbalanceDest = st.number_input("Receiver New Balance")

st.markdown("</div>", unsafe_allow_html=True)

# ===== Data =====
input_data = pd.DataFrame({
    'type': [type_trans],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest]
})

# ===== Prediction =====
st.write("")

if st.button("🚀 Predict Now"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("")

    if pred == 1:
        st.markdown(f"""
        <div class='card' style='border:1px solid red;'>
            <h2>🚨 FRAUD DETECTED</h2>
            <h3>Probability: {prob:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='card' style='border:1px solid #00ff99;'>
            <h2>✅ NORMAL TRANSACTION</h2>
            <h3>Probability: {prob:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)