import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("Predicción de Temperatura")
st.write("Esto es la primer prueba")

#  levantamos el df_test
with open('TP_INTEGRADOR/df_test.pkl', 'rb') as f_dftest:
        dataset_test = pickle.load(f_dftest)

st.write("Nuestro Dataset con las predicciones se ve asi")
st.write(dataset_test)



#fig = plt.figure(figsize=(15,15))

st.pyplot(dataset_test.plot(kind = "line", y = ['temp_min', 'model_ARIMA','predict_est']).figure)



def RMSE(predicted, actual):
    mse = (predicted - actual) ** 2
    rmse = np.sqrt(mse.sum() / mse.count())
    return rmse

st.write("Error de ARIMA")
RMSE(df_test['model_ARIMA'], df_test['temp_min'])
st.write("OLS")
RMSE(df_test['predict_est'], df_test['temp_min'])