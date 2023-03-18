import streamlit as st
import pickle

st.title("Predicción de Temperatura")
st.write("Esto es la primer prueba")

#  levantamos el df_test
with open('./df_test.pkl', 'rb') as f_dftest:
        dataset_test = pickle.load(f_dftest)




dataset_test.plot(kind = "line", y = ['temp_min', 'model_ARIMA','predict_est'])