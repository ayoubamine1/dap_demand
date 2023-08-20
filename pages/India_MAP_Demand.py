import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
prediction_style = {
        "font-size": "18px",
        "font-weight": "bold",
        "margin-top": "20px",
        "padding": "10px",
        "border": "2px solid #4CAF50",
        "border-radius": "8px",
         "color": "#214E3F",
        "background-color": "#E6F5E6",
        "box-shadow": "0px 2px 4px rgba(0, 0, 0, 0.2)"
    }


light_filename = 'Final model/India Final Model/lg_model.joblib'
random_filename = 'Final model/India Final Model/rf_model.joblib'
mate_model_filename = 'Final model/India Final Model/mate_model.joblib'

xgblags_filename = 'Final model/India Final Model/xgboost_model_withlags.joblib'
mata_lags_filename = 'Final model/India Final Model/meta_model_lags.joblib'


lgb_model = joblib.load(light_filename)
rf_model = joblib.load(random_filename)
meta_model = joblib.load(mate_model_filename)

xgb_model_with_lags  = joblib.load(xgblags_filename)
meta_model_lags = joblib.load(mata_lags_filename)
def get_user_input():
    st.write("Enter the predictor values:")

    # Group 1: Agricultural Production
    st.subheader("Agricultural Production")
    col1, col2 = st.columns(2)
    with col1:
        
        barley_production = st.slider("Barley Production (tons, lag 11)", min_value=1200000.0, max_value=1830000.0, value=1500000.0, step=1000.0)
        rice_paddy_production = st.slider("Rice Paddy Production (tons, lag 12)", min_value=179586400.0, max_value=296134200.0, value=250000000.0, step=1000.0)
        soybeans_production = st.slider("Soybeans Production (tons, lag 6)", min_value=4654700.0, max_value=14666000.0, value=10000000.0, step=1000.0)
        coffee_production = st.slider("Coffee Production (tons, lag 7)", min_value=1631852.0, max_value=3700231.0, value=2500000.0, step=1000.0)

    with col2:
        seed_cotton_unginned = st.slider("Seed cotton, unginned (tons, lag 6)", min_value=1478366.0, max_value=7070136.0, value=4500000.0, step=1000.0)
        sweet_potatoes = st.slider("Sweet Potatoes (tons, lag 12)", min_value=472422.0, max_value=803626.0, value=600000.0, step=1000.0)

    # Group 2: Area Harvested
    st.subheader("Area Harvested")
    col1, col2 = st.columns(2)
    with col1:
        barley_area_harvested = st.slider("Barley Area Harvested (hectares, lag 8)", min_value=575600.0, max_value=792800.0, value=700000.0, step=1000.0)
        rice_paddy_area_harvested = st.slider("Rice Paddy Area Harvested (hectares, lag 9)", min_value=41176100.0, max_value=45537400.0, value=44000000.0, step=1000.0)
        sugar_area_harvested = st.slider("Sugar Area Harvested (hectares, lag 4)", min_value=7323000.0, max_value=10300000.0, value=9000000.0, step=1000.0)

    with col2:
        beans_area_harvested = st.slider("Beans Area Harvested (hectares, lag 9)", min_value=5998038.0, max_value=15454290.0, value=10000000.0, step=1000.0)
        crop_land = st.slider("CropLand (hectares, lag 10)", min_value=0, max_value=10000000, value=5000000, step=1000)

    # Group 3: Climate and Energy
    st.subheader("Climate and Energy")
    temperature = st.slider("Temperature (Celsius, lag 2)", min_value=287.3971, max_value=300.6677, value=294.0824, step=0.01)
      

    # Group 4: Economic Factors
    st.subheader("Economic Factors")
    col1, col2 = st.columns(2)
    with col1:
        price = st.slider("Price (USD), lag 12", min_value=169619.5, max_value=0.0, value=100000.0, step=1000.0)
        oils_meals = st.slider("Oils & Meals (USD, lag 11)", min_value=36.79919, max_value=140.9597, value=50.0, step=0.01)
        grains = st.slider("Grains (USD, lag 0)", min_value=42.12176, max_value=156.6351, value=70.0, step=0.01)
    with col2:
        other_food = st.slider("Other Food (USD, lag 5)", min_value=48.94163, max_value=115.0557, value=80.0, step=0.01)
        fertilizers = st.slider("Fertilizers (USD, lag 2)", min_value=33.14611, max_value=256.0552, value=100.0, step=0.01)
        energy = st.slider("Energy (kWh, lag 4)", min_value=26.30775, max_value=173.4324, value=100.0, step=0.01)

    # Group 5: Exchange Rate and GDP
    st.subheader("Exchange Rate, GDP and Total Agriculture Investment ")
    col1, col2 = st.columns(2)
    with col1:
        exchange_rate = st.slider("Exchange Rate (lag 5)", min_value=41.34853, max_value=74.09957, value=60.0, step=0.01)
        gdp = st.slider("GDP (USD, lag 12)", min_value=3.086698, max_value=8.845756, value=5.0, step=0.01)
    with col2:
       
        total_agriculture_investment = st.slider("Total Agriculture Investment (USD, lag 11)", min_value=26484.2, max_value=155554.0, value=100000.0, step=1000.0)

    

    # Group 5: Time-related Factors
    st.header("Time-related Factors")
    col1, col2 = st.columns(2)
    months_index  = [0]*12
    month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                                             

    month = st.slider("Month", min_value=1, max_value=12, value=6, step=1)
    months_index[month-1] = 1

    quarter = 0
    
    if 10>month>=6 :
        quarter = 1
   
    # Lags of demand
    variables = {
        'price': price,
        'Barley_Production': barley_production,
        'Rice_Paddy_Production': rice_paddy_production,
        'Soybeans_Production': soybeans_production,
        'coffee_production': coffee_production,
        'Seed cotton, unginned': seed_cotton_unginned,
        'Sweet potatoes': sweet_potatoes,
        'Barley_AreaHarvested': barley_area_harvested,
        'Rice_Paddy_AreaHarvested': rice_paddy_area_harvested,
        'Sugar_AreaHarvested': sugar_area_harvested,
        'Beans_AreaHarvested': beans_area_harvested,
        'CropLand': crop_land,
        'temperature': temperature,
        'Energy': energy,
        'Oils & Meals': oils_meals,
        'Grains': grains,
        'Other Food': other_food,
        'Fertilizers': fertilizers,
        'ExchangeRate': exchange_rate,
        'GDP': gdp,
        'TotalAgricultureInvestment': total_agriculture_investment
    }
    
    for i in range(12):
        variables[month_name[i]] = months_index[i]
    variables['Quarter'] = quarter


    # Include lags of demand checkbox
    st.header("Imporve Model With Lags of Demand")
    include_lags = st.checkbox("Include Lags of Demand", value=False)
    lag_2 = lag_5 = lag_8 = lag_9 = lag_10 =  0
    lags  = {'lag_2': lag_2,
        'lag_5': lag_5,
        'lag_8': lag_8,
             'lag_9':lag_9,
        'lag_10':lag_10
    }

    if include_lags:
        lag_2 = st.slider("Lag 2 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_5 = st.slider("Lag 5 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_8 = st.slider("Lag 8 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_9 = st.slider("Lag 9 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_10 = st.slider("Lag 10 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        
        
        return [variables,lags]

    else :
        return [variables]
    

def make_prediction(user_input):
    prediction = None
    
    if len(user_input) == 1:
        
        data = pd.DataFrame([user_input[0]])
        lgb_error = lgb_model.predict(data)
        rf_error = rf_model.predict(data)

        meta_input = pd.DataFrame({
            'LR_error': lgb_error,
            'RF_error': rf_error,
        })
        prediction = meta_model.predict(meta_input)[0]
        
        
    
    else :
        data = pd.DataFrame([user_input[0]])
        lgb_error = lgb_model.predict(data)
        rf_error = rf_model.predict(data)
        data  = pd.concat([data,pd.DataFrame([user_input[1]])],axis=1)
        xgb_error = xgb_model_with_lags.predict(data)

        meta_input = pd.DataFrame({
            'LR_error': lgb_error ,
            'RF_error': rf_error,
            'XGB_error': xgb_error
        })

        prediction = meta_model_lags.predict(meta_input)[0]

    return prediction

def demand_wihtout_input(path):
     # Load your data here (replace 'FuturePredictions.csv' with your actual data file)
    final_data = pd.read_csv(path)
    final_data.index = pd.to_datetime(final_data['date'])

    # Your existing code for calculating X_error
    X_error = pd.DataFrame(index=final_data.index)
    error = final_data['Estimated Error']
    X_error['Prediction+Error'] = final_data['Predicted_Demand'] + error
    X_error['Prediction-Error'] = final_data['Predicted_Demand'] - error
    X_error['Prediction-Error'] = np.where(X_error['Prediction-Error'] < 0, 0, X_error['Prediction-Error'])
    X_error = X_error.groupby(X_error.index.year).sum()

    # Using a good color palette
    colors = ['#214E3F','#2F7F36','#69B132','#6FC02E','#9CCB27','#81C02C']

    # Create the figure using plotly.graph_objects
    fig1 = go.Figure()

     # Add the bar traces to the figure
    fig1.add_trace(go.Bar(
        y=X_error['Prediction-Error'],
        x=X_error.index,
        name='Min Demand Estimated',
        marker_color=colors[1],  # Second color from the palette
        marker_line_color='rgba(0,0,0,0)',
        text=round(X_error['Prediction-Error'], 2),
        textposition='auto'  # Add annotations to the bars
    ))

    # Add the 'Predicted Demand' bar with annotations
    fig1.add_trace(go.Bar(
        y=final_data.groupby(final_data.index.year).sum()['Predicted_Demand'],
        x=final_data.groupby(final_data.index.year).sum().index,
        name='Predicted Demand',
        marker_color=colors[0],  # First color from the palette
        text=round(final_data.groupby(final_data.index.year).sum()['Predicted_Demand'], 2),
        textposition='auto'  # Add annotations to the bars
    ))

    
   
    fig1.add_trace(go.Bar(
        y=X_error['Prediction+Error'],
        x=X_error.index,
        name='Max Demand Estimated',
        marker_color=colors[2],  # Third color from the palette
        marker_line_color='rgba(0,0,0,0)',
        text=round(X_error['Prediction+Error'], 2),
        textposition='auto'  # Add annotations to the bars
    ))

    # Update layout and axes labels for the first graph
    fig1.update_layout(
        title="Future Demand 2020-2029",
        yaxis_title="Demand",
        xaxis_title="Date",
        barmode='group',
        height=600, width=900,
    )
    # Remove gridlines from the x-axis and y-axis
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(showgrid=False)


    # Show the first graph
    st.plotly_chart(fig1)
    # Assuming you have 'y' and 'model_par' defined (from your existing code)
    # Create the second graph using plotly.express
    fig2 = px.line(final_data,y ='Predicted_Demand', title="Monthly Future Demand 2020-2029",
    color_discrete_sequence = colors)
    

    # Update layout and axes labels for the second graph
    fig2.update_layout(
        yaxis_title="Demand",
        xaxis_title="Date",
        height=600, width=800
    )

    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)

    # Show the second graph
    st.plotly_chart(fig2)

def main():
    st.markdown('<h1 >Predicting Agricultural Demand <img src="https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png" alt="Brazil Flag" width="40" style="vertical-align: middle;"></h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#E6B437">MAP in India</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Bienvenue sur la page de prédiction de la demande d'engrais agricoles en Inde.\
        Dans ce tableau de bord interactif, nous vous offrons la possibilité de plonger dans le monde de la \
        prédiction de la demande de MAP (phosphate monoammonique) dans le secteur agricole  \
       indien. Notre objectif est de vous fournir une perspective claire et approfondie sur les \
       tendances de la demande d'engrais, vous permettant ainsi de mieux comprendre les fluctuations saisonnières \
       et les facteurs qui influencent cette demande cruciale.\n Naviguez à travers les différentes sections de ce \
       tableau de bord pour explorer les résultats de nos analyses,les performances de nos modèles et les prévisions \
       futures de la demande de DAP. Nous vous invitons à découvrir l'intersection entre la science des données, l'agriculture \
       et la durabilité, tout en jetant un coup d'œil vers l'avenir de la prédiction de la demande agricole au Brésil.")
    with col2 :
        image = Image.open('india.jpg')
        st.image(image, caption='Technology empowering the agricultural sector')
    

    user_input = get_user_input()
    
    if st.button('Predict MAP Demand'):
        prediction = make_prediction(user_input)
        st.write(
        f'<div style="{"; ".join(f"{key}: {value}" for key, value in prediction_style.items())}">'
        f'Predicted Demand: {round(prediction,2)} tons'
        '</div>',
        unsafe_allow_html=True)

    st.title('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - FAO Data ')
    demand_wihtout_input('India Future Data/FuturePredictions.csv')
    st.title('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - Agro Climat Data')
    demand_wihtout_input('India Future Data/Agro_FuturePredictions.csv')
    

if __name__ == "__main__":
    main()
