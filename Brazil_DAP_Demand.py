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
from pages import India_MAP_Demand

prediction_style = {
        "font-size": "18px",
        "font-weight": "bold",
        "margin-top": "20px",
        "padding": "10px",
        "border": "2px solid #4CAF50",
        "border-radius": "8px",
        "background-color": "#E6F5E6",
        "box-shadow": "0px 2px 4px rgba(0, 0, 0, 0.2)"
    }



model_filename = 'Final model/xgboost_model.joblib'
light_filename = 'Final model/lg_model.joblib'
random_filename = 'Final model/rf_model.joblib'
mate_model_filename = 'Final model/mate_model.joblib'

xgblags_filename = 'Final model/xgboost_model_withlags.joblib'
mata_lags_filename = 'Final model/meta_model_lags.joblib'

xgb_model = joblib.load(model_filename)
lgb_model = joblib.load(light_filename)
rf_model = joblib.load(random_filename)
meta_model = joblib.load(mate_model_filename)

xgb_model_with_lags  = joblib.load(xgblags_filename)
meta_model_lags = joblib.load(mata_lags_filename)
def get_user_input():
    st.write("Enter the predictor values:")

    # Group 1: Agricultural Production and Area Harvested
    st.subheader("Agricultural Production and Area Harvested")

    # Use beta_columns to create two columns
    col1, col2 = st.columns(2)

    # Column 1: Sugar and Barley Production
    with col1:
        sugar = st.slider("Sugar Production (tons, lag 10)", min_value=16197710, max_value=38734080, value=30000000, step=100000)
        barley_production = st.slider("Barley Production (tons, lag 7)", min_value=186285, max_value=405615, value=300000, step=1000)
        rice_paddy_production = st.slider("Rice Paddy Production (tons, lag 0)", min_value=16977040, max_value=22466150, value=20000000, step=100000)
        wheat_production = st.slider("Wheat Production (tons, lag 10)", min_value=1725792, max_value=6834421, value=4000000, step=10000)
        beans_production = st.slider("Beans Production (tons, lag 5)", min_value=2191153, max_value=3486763, value=3000000, step=10000)

    # Column 2: Coffee and Seed Cotton Production
    with col2:
        coffee_production = st.slider("Coffee Production (tons, lag 2)", min_value=1631852, max_value=3700231, value=2000000, step=10000)
        seed_cotton_unginned = st.slider("Seed cotton, unginned (tons, lag 8)", min_value=1172992, max_value=6893340, value=4000000, step=10000)
        sweet_potatoes = st.slider("Sweet Potatoes (tons, lag 5)", min_value=444925, max_value=803626, value=600000, step=10000)
       

    # Group 2: Area Harvested
    st.subheader("Area Harvested")
    col1, col2 = st.columns(2)
    with col1:
        barley_area_harvested = st.slider("Barley Area Harvested (hectares, lag 6)", min_value=77452, max_value=156005, value=100000, step=1000)
        corn_area_harvested = st.slider("Corn Area Harvested (hectares, lag 3)", min_value=10585500, max_value=18253770, value=14000000, step=10000)
        rice_paddy_area_harvested = st.slider("Rice Paddy Area Harvested (hectares, lag 11)", min_value=1710063, max_value=3915855, value=3000000, step=10000)
    with col2: 
        wheat_area_harvested = st.slider("Wheat Area Harvested (hectares, lag 2)", min_value=1138687, max_value=2834945, value=2000000, step=10000)
        beans_area_harvested = st.slider("Beans Area Harvested (hectares, lag 3)", min_value=2587772, max_value=4332545, value=3500000, step=10000)

    # Group 3: Climate and Energy
    st.subheader("Climate")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (Celsius, lag 4)", min_value=0, max_value=50, value=25, step=1)  
    with col2:
        precip = st.slider("Precipitation (mm, lag 3)", min_value=12, max_value=105, value=50, step=1)
       

    # Group 4: Economic Factors
    st.subheader("Economic Factors")
    col1, col2 = st.columns(2)
    with col1:
        price = st.slider("Price (USD)", min_value=float(1), max_value=float(53000), value=float(2000), step=float(100))
        beverages = st.slider("Beverages (USD, lag 2)", min_value=float(30), max_value=float(150), value=float(50), step=0.01)
        oils_meals = st.slider("Oils & Meals (USD, lag 6)", min_value=float(30), max_value=float(150), value=float(50), step=0.01)
        fertilizers = st.slider("Fertilizers (USD, lag 8)", min_value=float(30), max_value=float(260), value=float(50), step=0.01)
       
    with col2:
        precious_metals = st.slider("Precious Metals (USD, lag 4)", min_value=float(20), max_value=float(160), value=float(50), step=0.01)
        exchange_rate = st.slider("Exchange Rate (lag 6)", min_value=float(1), max_value=float(4), value=float(2), step=0.01)
        energy = st.slider("Energy (USD, lag 7)", min_value=16, max_value=172, value=100, step=1)
        gdp = st.slider("GDP (USD, lag 0)", min_value=float(-4), max_value=float(8), value=float(2), step=0.01)
        total_agriculture_investment = st.slider("Total Agriculture Investment (USD, lag 12)", min_value=13940, max_value=73400, value=50000, step=100)

    # Group 5: Time-related Factors
    st.subheader("Time-related Factors")
    col1, col2 = st.columns(2)

    month = st.slider("Month", min_value=1, max_value=12, value=6, step=1)

    quarter = 0
    if 1<=month<4:
        quarter = 1      
    if 4<=month<7:
        quarter = 2
    if month>=7 :
        quarter = 0
   
    # Lags of demand
    variables = {
            'price': price,
            'sugar(tons)': sugar,
            'Barley_Production': barley_production,
            'Rice_Paddy_Production': rice_paddy_production,
            'Wheat_Production': wheat_production,
            'Beans_Production': beans_production,
            'coffee_production': coffee_production,
            'Seed cotton, unginned': seed_cotton_unginned,
            'Sweet potatoes': sweet_potatoes,
            'Barley_AreaHarvested': barley_area_harvested,
            'Corn_AreaHarvested': corn_area_harvested,
            'Rice_Paddy_AreaHarvested': rice_paddy_area_harvested,
            'Wheat_AreaHarvested': wheat_area_harvested,
            'Beans_AreaHarvested': beans_area_harvested,
            'temperature': temperature+273.14,
            'precip': precip,
            'Energy': energy,
            'Beverages': beverages,
            'Oils & Meals': oils_meals,
            'Fertilizers': fertilizers,
            'Precious Metals': precious_metals,
            'ExchangeRate': exchange_rate,
            'GDP': gdp,
            'TotalAgricultureInvestment': total_agriculture_investment,
            'Month': month,
            'Quarter': quarter
        }

    # Include lags of demand checkbox
    st.subheader("Imporve Model With Lags of Demand")
    include_lags = st.checkbox("Include Lags of Demand", value=False)
    lag_5 = lag_3 = lag_12 = 0
    lags  = {'lag_5': lag_5,
        'lag_3': lag_3,
        'lag_12': lag_12
    }

    if include_lags:
        lag_5 = st.slider("Lag 5 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_3 = st.slider("Lag 3 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        lag_12 = st.slider("Lag 12 of Demand", min_value=0, max_value=100000, value=0, step=1000)
        return [variables,lags]

    else :
        return [variables]
    

def make_prediction(user_input):
    prediction = None
   
    if len(user_input) == 1:
        data = pd.DataFrame([user_input[0]])
        xgb_error = xgb_model.predict(data)
        lgb_error = lgb_model.predict(np.array(data))
        rf_error = rf_model.predict(data)

        meta_input = pd.DataFrame({
            'LR_error': lgb_error,
            'RF_error': rf_error,
            'XGB_error': xgb_error
        })
        prediction = meta_model.predict(meta_input)[0]
        
    
    else :
        data = pd.DataFrame([user_input[0]])
        lgb_error = lgb_model.predict(np.array(data))
        rf_error = rf_model.predict(data)
        data  = pd.concat([data,pd.DataFrame([user_input[1]])],axis=1)
        xgb_error = xgb_model_with_lags.predict(data)

        meta_input = pd.DataFrame({
            'LR_error': lgb_error,
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
        title="Future Demand 2020-2030",
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
    fig2 = px.line(final_data,y ='Predicted_Demand', title="Monthly Future Demand 2020-2030",
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
    
    st.header('Future Demand Prediction with Predictor Variables')
    user_input = get_user_input()
    
    if st.button("Predict DAP Demand", key="predict_button", type="primary"):
        prediction = make_prediction(user_input)
        st.markdown(
        f'<div style="{"; ".join(f"{key}: {value}" for key, value in prediction_style.items())}">'
        f'Predicted Demand: {prediction} tons'
        '</div>',
        unsafe_allow_html=True)

        

    st.header('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - FAO Data ')
    demand_wihtout_input('Brazil Future Data/FuturePredictions.csv')
    st.header('Predicting Agricultural Demand for DAP in Brazil - Hybrid Approach - Agro Climat Data')
    demand_wihtout_input('Brazil Future Data/Agro_FuturePredictions.csv')
    

if __name__ == "__main__":
    # Add a navigation sidebar to switch between pages
    st.sidebar.title("Predicting Agricultural Demand")
    page_options = ["Brazil DAP Demand", "India MAP Demand"]
    selected_page = st.sidebar.selectbox("Select Page", page_options)
    if selected_page == "India MAP Demand":
        India_MAP_Demand.main()
    else :
        main()
