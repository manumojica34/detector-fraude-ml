import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el escalador
modelo = joblib.load('modelo_fraude_rf.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')

st.title("Detector de Fraude Financiero")
st.write("Ingresa los datos de la transacción para evaluarla con Random Forest.")

# Entradas del usuario
monto = st.number_input("Monto de la transacción ($)", min_value=0.0, value=1000.0)
saldo_orig = st.number_input("Saldo original de la cuenta ($)", min_value=0.0, value=5000.0)
nuevo_saldo = st.number_input("Nuevo saldo de la cuenta ($)", min_value=0.0, value=4000.0)
tipo = st.selectbox("Tipo de transacción", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])

if st.button("Evaluar Transacción"):
    # Replicar las variables de tu modelo
    balance_diff_orig = saldo_orig - nuevo_saldo
    monto_scaled = scaler_amount.transform([[monto]])[0][0]

    # Armar la tabla con las mismas columnas y orden de tu entrenamiento
    datos = pd.DataFrame([{
        'amount_scaled': monto_scaled,
        'balance_diff_orig': balance_diff_orig,
        'balance_diff_dest': 0.0,
        'step_scaled': 0.0,
        'type_CASH_OUT': 1 if tipo == "CASH_OUT" else 0,
        'type_DEBIT': 1 if tipo == "DEBIT" else 0,
        'type_PAYMENT': 1 if tipo == "PAYMENT" else 0,
        'type_TRANSFER': 1 if tipo == "TRANSFER" else 0
    }])

    # Predicción
    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0][1] * 100

    if prediccion == 1:
        st.error(f"🚨 ALERTA DE FRAUDE (Probabilidad: {probabilidad:.2f}%)")
    else:
        st.success(f"✅ Transacción Legítima (Probabilidad de fraude: {probabilidad:.2f}%)")