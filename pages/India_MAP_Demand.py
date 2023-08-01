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



light_filename = 'Final model/lg_model.joblib'
random_filename = 'Final model/rf_model.joblib'
mate_model_filename = 'Final model/mate_model.joblib'

xgblags_filename = 'Final model/xgboost_model_withlags.joblib'
mata_lags_filename = 'Final model/meta_model_lags.joblib'


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
        
        barley_production = st.slider("Barley Production (tons, lag 11)", min_value=1.200000e+06, max_value=1.830000e+06, value=1.500000e+06, step=1000)
        rice_paddy_production = st.slider("Rice Paddy Production (tons, lag 12)", min_value=1.795864e+08, max_value=2.961342e+08, value=2.500000e+08, step=1000)
        soybeans_production = st.slider("Soybeans Production (tons, lag 6)", min_value=4.654700e+06, max_value=1.466600e+07, value=1.000000e+07, step=1000)
        coffee_production = st.slider("Coffee Production (tons, lag 7)", min_value=1.631852e+06, max_value=3.700231e+06, value=2.500000e+06, step=1000)

    with col2:
        seed_cotton_unginned = st.slider("Seed cotton, unginned (tons, lag 6)", min_value=1.478366e+06, max_value=7.070136e+06, value=4.500000e+06, step=1000)
        sweet_potatoes = st.slider("Sweet Potatoes (tons, lag 12)", min_value=4.724220e+05, max_value=8.036260e+05, value=6.000000e+05, step=1000)

    # Group 2: Area Harvested
    st.subheader("Area Harvested")
    col1, col2 = st.columns(2)
    with col1:
        barley_area_harvested = st.slider("Barley Area Harvested (hectares, lag 8)", min_value=5.756000e+05, max_value=7.928000e+05, value=7.000000e+05, step=1000)
        rice_paddy_area_harvested = st.slider("Rice Paddy Area Harvested (hectares, lag 9)", min_value=4.117610e+07, max_value=4.553740e+07, value=4.400000e+07, step=1000)
        sugar_area_harvested = st.slider("Sugar Area Harvested (hectares, lag 4)", min_value=7.323000e+06, max_value=1.030000e+07, value=9.000000e+06, step=1000)

    with col2:
        beans_area_harvested = st.slider("Beans Area Harvested (hectares, lag 9)", min_value=5.998038e+06, max_value=1.545429e+07, value=1.000000e+07, step=1000)
        crop_land = st.slider("CropLand (hectares, lag 10)", min_value=0, max_value=1e7, value=5e6, step=1000)

    # Group 3: Climate and Energy
    st.subheader("Climate and Energy")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (Celsius, lag 2)", min_value=2.873971e+02, max_value=3.006677e+02, value=2.940824e+02, step=0.01)
        

    # Group 4: Economic Factors
    st.subheader("Economic Factors")
    col1, col2 = st.columns(2)
    with col1:
        price = st.slider("Price (USD), lag 12", min_value=1.696195e+05, max_value=0.000000e+00, value=1.000000e+05, step=1000)
        oils_meals = st.slider("Oils & Meals (USD, lag 11)", min_value=3.679919e+01, max_value=1.409597e+02, value=5.000000e+01, step=0.01)
        grains = st.slider("Grains (USD, lag 0)", min_value=4.212176e+01, max_value=1.566351e+02, value=7.000000e+01, step=0.01)
    with col2:
        other_food = st.slider("Other Food (USD, lag 5)", min_value=4.894163e+01, max_value=1.150557e+02, value=8.000000e+01, step=0.01)
        fertilizers = st.slider("Fertilizers (USD, lag 2)", min_value=3.314611e+01, max_value=2.560552e+02, value=1.000000e+02, step=0.01)
        energy = st.slider("Energy (usd, lag 4)", min_value=2.630775e+01, max_value=1.734324e+02, value=1.000000e+02, step=0.01)

    # Group 5: Exchange Rate and GDP
    st.subheader("Exchange Rate, GDP and Total Agriculture Investment")
    col1, col2 = st.columns(2)
    with col1:
        exchange_rate = st.slider("Exchange Rate (lag 5)", min_value=4.134853e+01, max_value=7.409957e+01, value=6.000000e+01, step=0.01)
        gdp = st.slider("GDP (USD, lag 12)", min_value=3.086698e+00, max_value=8.845756e+00, value=5.000000e+00, step=0.01)
        
    with col2:
        total_agriculture_investment = st.slider("Total Agriculture Investment (USD, lag 11)", min_value=2.648420e+04, max_value=1.555540e+05, value=1.000000e+05, step=1000)

    

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
        "price": price,
        "barley_production": barley_production,
        "rice_paddy_production": rice_paddy_production,
        "soybeans_production": soybeans_production,
        "coffee_production": coffee_production,
        "seed_cotton_unginned": seed_cotton_unginned,
        "sweet_potatoes": sweet_potatoes,
        "barley_area_harvested": barley_area_harvested,
        "rice_paddy_area_harvested": rice_paddy_area_harvested,
        "sugar_area_harvested": sugar_area_harvested,
        "beans_area_harvested": beans_area_harvested,
        "corn_area_harvested": corn_area_harvested,
        "soybeans_area_harvested": soybeans_area_harvested,
        "temperature": temperature,
        "energy": energy,
        "oils_meals": oils_meals,
        "grains": grains,
        "other_food": other_food,
        "fertilizers": fertilizers,
        "exchange_rate": exchange_rate,
        "gdp": gdp,
        "total_agriculture_investment": total_agriculture_investment
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
            'LR_error': xgb_error,
            'RF_error': lgb_error,
        })
        prediction = meta_model.predict(meta_input)[0]
    
    else :
        data = pd.DataFrame([user_input[0]])
        lgb_error = lgb_model.predict(data)
        rf_error = rf_model.predict(data)
        data  = pd.concat([data,pd.DataFrame([user_input[1]])],axis=1)
        xgb_error = xgb_model_with_lags.predict(data)

        meta_input = pd.DataFrame({
            'LR_error': xgb_error,
            'RF_error': lgb_error,
            'XGB_error': rf_error
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
    colors = px.colors.qualitative.Pastel2

    # Create the figure using plotly.graph_objects
    fig1 = go.Figure()

    # Add the bar traces to the figure
    fig1.add_trace(go.Bar(
        y=X_error['Prediction-Error'],
        x=X_error.index,
        name='Min Demand Estimated',
        marker_color=colors[1],  # Second color from the palette
        text=round(X_error['Prediction-Error'], 2),
        textposition='auto',  # Add annotations to the bars
        showgrid=False
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
        text=round(X_error['Prediction+Error'], 2),
        textposition='auto'  # Add annotations to the bars
    ))
    fig1.update_xaxes(showgrid=False)  # Remove gridlines for the x-axis
    fig1.update_yaxes(showgrid=False)  # Remove gridlines for the y-axis

    # Update layout and axes labels for the first graph
    fig1.update_layout(
        title="Future Demand 2020-2029",
        yaxis_title="Demand",
        xaxis_title="Date",
        barmode='group',
        height=600, width=1000
    )

    # Show the first graph
    st.plotly_chart(fig1)

    # Assuming you have 'y' and 'model_par' defined (from your existing code)
    # Create the second graph using plotly.express
    fig2 = px.line(final_data,y ='Predicted_Demand', title="Monthly Future Demand 2020-2029",
    color_discrete_sequence =px.colors.qualitative.Pastel2)
    

    # Update layout and axes labels for the second graph
    fig2.update_layout(
        yaxis_title="Demand",
        xaxis_title="Date",
        height=600, width=1000
    )
    fig2.update_xaxes(showgrid=False)  # Remove gridlines for the x-axis
    fig2.update_yaxes(showgrid=False)  # Remove gridlines for the y-axis

    # Show the second graph
    st.plotly_chart(fig2)


def main():
    st.title('Predicting Agricultural Demand for MAP in India')
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
    

    user_input = get_user_input()
    if st.button("Predict  DAP Demand"):
        prediction = make_prediction(user_input)

        st.write("Predicted Demand:", prediction)

    st.title('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - FAO Data ')
    demand_wihtout_input('Brazil Future Data/FuturePredictions.csv')
    st.title('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - Agro Climat Data')
    demand_wihtout_input('Brazil Future Data/Agro_FuturePredictions.csv')
    

if __name__ == "__main__":
    main()
