import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

st.title("Predicción de Temperatura")
st.write("Esto es la primer prueba")

#  levantamos el df_test
with open('TP_INTEGRADOR/df_test.pkl', 'rb') as f_dftest:
        dataset_test = pickle.load(f_dftest)

dataset_test.plot(kind = "line", y = ['temp_min', 'model_ARIMA','predict_est'])