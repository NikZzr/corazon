import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------------
# Entrenamiento del modelo
# ----------------------
# Dataset de ejemplo (edad, colesterol, problemas cardiacos)
data = {
    "edad": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "colesterol": [180, 190, 200, 210, 220, 240, 260, 280, 300, 320],
    "problema_cardiaco": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["edad", "colesterol"]]
y = df["problema_cardiaco"]

# Entrenar modelo simple
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------
# Interfaz en Streamlit
# ----------------------
st.title("🫀 Predictor de problemas cardiacos")
st.subheader("Elaborado por ® UNAB2025")

# Entradas del usuario
edad = st.number_input("Ingrese la edad del paciente:", min_value=1, max_value=120, step=1)
colesterol = st.number_input("Ingrese el colesterol del paciente (mg/dL):", min_value=100, max_value=400, step=2)

# Botón de predicción
if st.button("Predecir"):
    features = np.array([[edad, colesterol]])
    pred = model.predict(features)[0]

    if pred == 0:
        st.success("✅ El paciente NO sufrirá del corazón.")
        st.image("https://www.shutterstock.com/image-photo/young-handsome-hicpanic-man-smiling-260nw-2527368779.jpg", width=300)
    else:
        st.error("⚠️ El paciente SUFRIRÁ del corazón.")
        st.image("https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg", width=400)
