# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================================
# 1. CONFIGURACIÓN INICIAL
# ======================================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ No se encontró la clave GEMINI_API_KEY en el archivo .env")

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)

st.title("📊 Predicción de Deserción Estudiantil")
st.write("Este sistema predice el riesgo de deserción de un estudiante y da un análisis general del dataset.")

# ======================================
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ======================================
file_path = "higher_education.csv"
df = pd.read_csv(file_path)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convertir Target: Dropout=1 (deserta), Enrolled/Graduate=0
df["Deserta"] = df["Target"].apply(lambda t: 1 if t.lower() == "dropout" else 0)

# Variables predictoras
X = df.drop(["Target", "Deserta"], axis=1)
y = df["Deserta"]

# Codificación de variables categóricas
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Balanceo con SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ======================================
# 3. ENTRENAMIENTO DE MODELOS
# ======================================
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluación
results = []
for model, name in [(rf, "Random Forest"), (dt, "Decision Tree")]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else None
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1, auc))
    st.write(f"🔍 {name} -> Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc}")

# Elegir mejor modelo
best = max(results, key=lambda x: (x[3] if not np.isnan(x[3]) else 0))
model = rf if best[0] == "Random Forest" else dt
st.success(f"✅ Modelo final seleccionado: {type(model).__name__}")

# ======================================
# 4. ANÁLISIS GENERAL CON GEMINI
# ======================================
if st.button("🧠 Analizar dataset general"):
    prompt_general = f"""
    Analiza los datos de educación superior y su relación con la deserción estudiantil.
    Explica en español:
    - Qué variables influyen más en la deserción
    - Qué perfiles parecen más vulnerables
    - Posibles patrones interesantes

    Columnas: {', '.join(df.columns.tolist())}

    Ejemplo de registros:
    {df.head(5).to_string(index=False)}
    """
    analisis_general = chat.invoke(prompt_general).content
    st.subheader("Análisis General del Dataset")
    st.write(analisis_general)

# ======================================
# 5. FORMULARIO DE PREDICCIÓN INDIVIDUAL
# ======================================
st.subheader("🔮 Predicción Individual de Deserción")
with st.form("form_estudiante"):
    new_data = {}
    for col in X.columns:
        if col in label_encoders:
            val = st.selectbox(col, options=label_encoders[col].classes_)
        else:
            val = st.number_input(col, value=0.0)
        new_data[col] = val
    submitted = st.form_submit_button("Predecir")

if submitted:
    new_df = pd.DataFrame([new_data])
    proba = model.predict_proba(new_df)[0][1] if len(model.classes_) > 1 else model.predict(new_df)[0]
    pred = 1 if proba >= 0.5 else 0
    conclusion = f"⚠️ Riesgo alto de deserción: {proba*100:.2f}%" if pred==1 else f"✅ Riesgo bajo de deserción: {(1-proba)*100:.2f}%"
    st.write(conclusion)

    # Análisis con Gemini
    prompt_individual = f"""
    Actúa como analista educativo.
    El sistema predijo una probabilidad de deserción del {proba*100:.2f}%.
    Basándote en los siguientes datos del estudiante:

    {new_data}

    Explica:
    - Posibles razones de la predicción
    - Recomendaciones específicas para reducir el riesgo
    """
    analisis_individual = chat.invoke(prompt_individual).content
    st.subheader("🧩 Análisis Individual generado por Gemini")
    st.write(analisis_individual)
