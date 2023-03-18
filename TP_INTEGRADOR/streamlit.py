import streamlit as st
import pickle
import seaborn as sns

st.title("Predicción de Temperatura")
st.write("Esto es la primer prueba")

#  levantamos el df_test
with open('TP_INTEGRADOR/df_test.pkl', 'rb') as f_dftest:
        dataset_test = pickle.load(f_dftest)

grafico = plt.figure(figsize=(10,4))
dataset_test.plot(kind = "line", y = ['temp_min', 'model_ARIMA','predict_est'])

st.pyplot(grafico)