import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
import joblib


st.sidebar.success("Select Any Page from here")

st.title('Predicting Agricultural Demand for DAP in Brazil')
col1, col2 = st.columns(2)
with col1:
    st.markdown('The agricultural sector plays a crucial role in the global economy and is intricately \
    linked to various factors such as production, population, weather conditions,\
     prices, economic conditions, and government policies. In this project, our main objective is to predict \
    the agricultural demand for Di-Ammonium Phosphate (DAP) in Brazil. By analyzing historical data and utilizing \
     advanced modeling techniques, we aim to provide accurate predictions that can aid in decision-making and resource \
    planning for agricultural stakeholders in Brazil. Understanding the factors influencing agricultural demand is essential \
     for optimizing production, ensuring food security, and driving sustainable growth in the agricultural sector.')
with col2 :
    image = Image.open('head1.jpg')
    st.image(image, caption='Technology empowering the agricultural sector')
    

st.header("Agriculture Production")

col1, col2 = st.columns(2)
with col1:
    st.text("Barley Production")
    pro = st.text_input('Tonnes',330374)


with col1:
    st.text("Barley Area Harvested")
    surf = st.text_input('mm2',101370 )


st.header("Weather Conditions")

col1, col2 = st.columns(2)
with col1:
    st.text("Temperature")
    temp = st.slider('kelvin', 273, 313,273+25 )


with col1:
    st.text("Precipitation")
    precip = st.slider('', 10, 150,60)

st.header("Economic Situation")

col1, col2 = st.columns(2)
with col1:
    st.text("Gross domestic product")
    gdp = st.text_input('gdp', 1.783666762)
    

st.header("Government policies")

col1, col2 = st.columns(2)
with col1:
    st.text("Total Agriculture Investment")
    tai = st.text_input('', 87.0105705167065)


st.header("Previous Demand")


col1, col2 = st.columns(2)

with col1:
    lag_1 =  st.text_input('month 1',42089.47 )
    lag_2 = st.text_input('month 2',9743.11 )

with col2:
    lag_3 = st.text_input('month 10',33152.48 )
    lag_4 = st.text_input('month 11', 32582.39)




scaler_X = joblib.load('../Final model/scaler_X.save')
scaler_y  = joblib.load('../Final model/scaler_y.save')
model = load_model('../Final model/my_model.h5')




if st.button("Predict  DAP Demand"):
    X_future =  np.array([float(pro),float(surf),float(temp)
             ,float(precip),float(gdp),float(tai),
             float(lag_1),float(lag_2),float(lag_3),float(lag_4)])
    X_future_scaled_ = scaler_X.transform(X_future.reshape(1,-1))
    X_future_reshaped_ = np.reshape(X_future_scaled_,
                                    (X_future_scaled_.shape[0], X_future_scaled_.shape[1], 1))
    
    y_pred_scaled_ = model.predict(X_future_reshaped_)
    y_pred_ = scaler_y.inverse_transform(y_pred_scaled_)
    
    st.text('the quanity demanded is: '+ str(float(y_pred_))+' Tonnes')

