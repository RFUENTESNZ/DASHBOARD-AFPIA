
# DASHBOARD AFP IA REESTRUCTURADO CON FLUJO INTEGRADO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Configuración inicial
st.set_page_config(page_title="Dashboard AFP IA", layout="wide", page_icon="📊")
st.title("📊 Dashboard AFP con Inteligencia Artificial Real")
st.caption("Simulador y análisis de acceso a beneficios previsionales, con modelos predictivos e interpretación automática.")

# Simular base limpia
np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'edad': np.random.randint(50, 90, n),
    'meses_cotizados': np.random.randint(0, 500, n),
    'sexo': np.random.choice(['F', 'M'], n),
    'pensionado': np.random.choice([0, 1], n)
})

def aplica_regla(row):
    if row['edad'] >= 65 and row['pensionado'] == 1:
        if row['sexo'] == 'F' and row['meses_cotizados'] >= 120:
            return 1
        if row['sexo'] == 'M' and row['meses_cotizados'] >= 240:
            return 1
    return 0

df['consultara_beneficio'] = df.apply(aplica_regla, axis=1)

# Entrenar modelo IA
df_model = df.copy()
df_model['sexo'] = df_model['sexo'].map({'F': 0, 'M': 1})
X = df_model[['edad', 'meses_cotizados', 'sexo', 'pensionado']]
y = df_model['consultara_beneficio']
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)
importancias = modelo.feature_importances_

# Detección de casos atípicos
outlier_model = IsolationForest(contamination=0.02, random_state=42)
df['outlier'] = outlier_model.fit_predict(X)
df['posible_error'] = df['outlier'] == -1

# --------------------- 1. VISTA GENERAL ---------------------
st.subheader("🔍 Análisis general de la base de afiliados")
st.markdown("Explora cuántas personas califican al beneficio según sus características. Usa los filtros para analizar diferentes segmentos.")

sexo_op = st.selectbox("Filtrar por sexo", options=["Todos", "F", "M"])
edad_rango = st.slider("Rango de edad", min_value=50, max_value=90, value=(60, 75))
solo_pensionados = st.checkbox("Solo pensionados", value=True)

filtro = (df['edad'] >= edad_rango[0]) & (df['edad'] <= edad_rango[1])
if sexo_op != "Todos":
    filtro &= (df['sexo'] == sexo_op)
if solo_pensionados:
    filtro &= (df['pensionado'] == 1)

df_filtrado = df[filtro]
col1, col2, col3 = st.columns(3)
col1.metric("👥 Personas filtradas", len(df_filtrado))
col2.metric("✅ Califican", int(df_filtrado['consultara_beneficio'].sum()))
col3.metric("❌ No califican", int(len(df_filtrado) - df_filtrado['consultara_beneficio'].sum()))

graf = px.pie(df_filtrado, names='consultara_beneficio', title="Distribución según beneficio", color_discrete_sequence=['red', 'green'], labels={0: "No", 1: "Sí"})
st.plotly_chart(graf, use_container_width=True)
st.dataframe(df_filtrado.reset_index(drop=True), use_container_width=True)

# --------------------- 2. SIMULADOR PERSONAL ---------------------
st.subheader("🧠 Simulador IA personal")
st.markdown("Completa los datos de una persona y el modelo te dirá si calificaría al beneficio.")

with st.form("simulador"):
    c1, c2, c3 = st.columns(3)
    edad = c1.number_input("Edad", 18, 100, 63)
    meses = c2.number_input("Meses Cotizados", 0, 500, 100)
    sexo = c3.selectbox("Sexo", ["F", "M"])
    pensionado = st.selectbox("¿Está pensionado?", ["Sí", "No"])
    enviar = st.form_submit_button("Predecir")

if enviar:
    input_df = pd.DataFrame([{
        'edad': edad,
        'meses_cotizados': meses,
        'sexo': 0 if sexo == 'F' else 1,
        'pensionado': 1 if pensionado == 'Sí' else 0
    }])
    pred = modelo.predict(input_df)[0]
    prob = modelo.predict_proba(input_df)[0][1]

    st.success(f"✅ {'Recibiría' if pred==1 else 'No recibiría'} el beneficio.")
    st.info(f"Probabilidad estimada: {round(prob*100, 2)}%")

    if pred == 0:
        if edad < 65:
            st.warning(f"📌 Debes esperar {65 - edad} años más.")
        if pensionado != 'Sí':
            st.warning("📌 Debes estar pensionado.")
        if sexo == 'F' and meses < 120:
            st.warning(f"📌 Te faltan {120 - meses} meses de cotización.")
        if sexo == 'M' and meses < 240:
            st.warning(f"📌 Te faltan {240 - meses} meses de cotización.")
    st.markdown(f"🧠 *Copiloto IA:* Con {edad} años y {meses} meses cotizados, {'pensionado' if pensionado == 'Sí' else 'aún no pensionado'}, tu probabilidad es de {round(prob*100)}%. {('¡Felicidades!' if pred==1 else 'Estás cerca, sigue cotizando.')}")

# --------------------- 3. SIMULACIÓN FUTURA ---------------------
st.subheader("🔮 Proyección futura con IA")
st.markdown("¿Qué pasaría si las personas que hoy no califican siguen cotizando?")

proy = []
df_no = df[df['consultara_beneficio'] == 0].copy()
for extra in [0, 6, 12, 24, 36]:
    temp = df_no.copy()
    temp['meses_cotizados'] += extra
    temp['simulado'] = temp.apply(aplica_regla, axis=1)
    proy.append((extra, temp['simulado'].sum()))

df_proy = pd.DataFrame(proy, columns=['meses_extra', 'personas_que_califican'])
graf_proy = px.line(df_proy, x='meses_extra', y='personas_que_califican', markers=True,
                    title="Beneficiarios potenciales si siguen cotizando")
st.plotly_chart(graf_proy, use_container_width=True)

# --------------------- 4. CASOS ATÍPICOS ---------------------
st.subheader("🚨 Casos detectados por IA como apelables o atípicos")
df_out = df[df['posible_error']]
st.info(f"Detectamos {len(df_out)} casos atípicos. Revisa si alguno podría calificar por error.")
st.dataframe(df_out[['edad', 'meses_cotizados', 'sexo', 'pensionado', 'consultara_beneficio']])

# --------------------- 5. EXPLICACIÓN IA ---------------------
st.subheader("📈 ¿Qué influye más en la decisión del modelo?")
df_imp = pd.DataFrame({'feature': ['edad', 'meses_cotizados', 'sexo', 'pensionado'], 'importancia': importancias}).sort_values(by='importancia')
graf_imp = px.bar(df_imp, x='importancia', y='feature', orientation='h', title="Importancia de variables")
st.plotly_chart(graf_imp, use_container_width=True)

