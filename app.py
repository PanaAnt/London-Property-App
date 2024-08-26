from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import re
import os
import math
import time 
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import streamlit as st 
import numpy as np
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import json
# Redefining np.float_ to np.float64 since prophet does not work well with np.float
np.float_ = np.float64
from prophet import Prophet

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#App 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

st.title("**London Property Tool**:house_with_garden::hammer_and_wrench:")
st.divider()
st.subheader("About the tool")
st.write("This tool is **NOT** offering financial advice **AT ALL**. It is a tool that aims to provide insights into the current London property market with a focus on analysing the different Boroughs within the city. The implementation of the Buy-To-Let Calculator is due to my personal interest of investing into property in the future and combining that with visuals and a LIVE data source provides me with a significant level of information for insights into potential investment areas. For the ***user***, it could also produce meaningful insights for you as well.")
df = pd.read_csv('data//UK-HPI-full-file-2024-06.csv')

# List of London Boroughs
london_boroughs = [
    'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent',
    'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich',
    'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering',
    'Hillingdon', 'Hounslow', 'Islington', 'Kensington and Chelsea',
    'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
    'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
    'Waltham Forest', 'Wandsworth', 'City of Westminster'
]

# Borough Filter
df_london = df[df['RegionName'].isin(london_boroughs)].copy()
df_london['Date'] = pd.to_datetime(df_london['Date'], format='%d/%m/%Y')


# Sidebar Navigation
st.sidebar.title("Navigation")
sections = [
    "Home",
    "Property Prices & Sales Volume", 
    "House Price Predictions", 
    "Forecast Metrics", 
    "Buy-To-Let Mortgage Calculator", 
    "Average Rent & Map", 
    "RightMove **LIVE** Rents Up for Sale",
]
selected_section = st.sidebar.radio("Go to:", sections)

#GRAPHS FOR PROPERTY PRICES ACROSS 30 YEARS------------------------------------------------------------------------------------------------------------------------------
# Section 1: Property Prices & Sales Volume
def show_property_prices_sales_volume():
    st.title("London Borough Property-type: Prices & Sales Volume")
    
    london_boroughs = [
        'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent',
        'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich',
        'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering',
        'Hillingdon', 'Hounslow', 'Islington', 'Kensington and Chelsea',
        'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
        'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
        'Waltham Forest', 'Wandsworth', 'City of Westminster'
    ]

    df_london = pd.read_csv('data//UK-HPI-full-file-2024-06.csv')  
    df_london['Date'] = pd.to_datetime(df_london['Date'], format='%d/%m/%Y')

    borough = st.selectbox("Select a London Borough:", london_boroughs)
    df_borough = df_london[df_london['RegionName'] == borough]

    st.subheader(f"Property Prices in {borough}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_borough['Date'], df_borough['AveragePrice'], label='Average Price', marker='.')
    ax.plot(df_borough['Date'], df_borough['DetachedPrice'], label='Detached Price', marker='.')
    ax.plot(df_borough['Date'], df_borough['SemiDetachedPrice'], label='Semi-Detached Price', marker='.')
    ax.plot(df_borough['Date'], df_borough['TerracedPrice'], label='Terraced Price', marker='.')
    ax.plot(df_borough['Date'], df_borough['FlatPrice'], label='Flat Price', marker='.')

    ax.set_title(f"Price Trends in {borough}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (£)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader(f"Sales Volume in {borough}")
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df_borough['Date'], df_borough['SalesVolume'], label='Sales Volume', color='steelblue', marker='o')
    ax.set_title(f"Sales Volume Trends in {borough}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Volume")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Section 2: House Price Predictions
def show_house_price_predictions():
    st.title("House Price Predictions 📈")

    df = pd.read_csv('data//UK-HPI-full-file-2024-06.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    london_boroughs = [
        'City of London', 'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent',
        'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich',
        'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering',
        'Hillingdon', 'Hounslow', 'Islington', 'Kensington and Chelsea',
        'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
        'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
        'Waltham Forest', 'Wandsworth', 'City of Westminster'
    ]

    selected_borough = st.selectbox("Select a Borough", london_boroughs)

    borough_data = df[df['RegionName'] == selected_borough]

    if not borough_data.empty:
        prophet_data = borough_data[['Date', 'AveragePrice']].rename(columns={'Date': 'ds', 'AveragePrice': 'y'})

        model = Prophet()
        model.fit(prophet_data)

        future = model.make_future_dataframe(periods=36, freq='M')
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(15, 8))
        model.plot(forecast, ax=ax)
        ax.set_title(f"Forecast for {selected_borough}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price")
        st.pyplot(fig)
    else:
        st.write("No data available for the selected borough.")

# Section 3: Forecast Metrics
def show_forecast_metrics():
    st.title("Forecast Metrics by Borough :1234:")
    st.subheader("This section is for a more in-depth look at the performance of the 'prophet' forecasting model.")

    df_london = pd.read_csv('data//UK-HPI-full-file-2024-06.csv')
    forecast_df = pd.read_csv('data//london_boroughs_forecast.csv')

    df_london['Date'] = pd.to_datetime(df_london['Date'], format='%d/%m/%Y')
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    merged_df = pd.merge(forecast_df, df_london, left_on=['Borough', 'ds'], right_on=['RegionName', 'Date'], how='inner')
    merged_df.drop(columns=['Date', 'RegionName'], inplace=True)
    cleaned_df = merged_df.dropna(subset=['AveragePrice', 'yhat'])

    rmse_by_borough = cleaned_df.groupby('Borough').apply(lambda x: np.sqrt(mean_squared_error(x['AveragePrice'], x['yhat'])))
    mae_by_borough = cleaned_df.groupby('Borough').apply(lambda x: mean_absolute_error(x['AveragePrice'], x['yhat']))

    metrics_by_borough = pd.DataFrame({
        'RMSE (in 1000s)': rmse_by_borough,
        'MAE (in 1000s)': mae_by_borough
    })

    st.warning("**-RMSE** (Root Mean Square Error) measures the average magnitude of the error between predicted and actual values, giving higher weight to larger errors due to squaring them before averaging.")
    st.warning("**-MAE** (Mean Absolute Error) measures the average absolute difference between predicted and actual values, treating all errors equally without amplifying larger ones.")

    st.write("### RMSE and MAE by Borough Table")
    st.dataframe(metrics_by_borough)

    #st.write("### RMSE and MAE Bar Charts")
    #fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    #axes[0].bar(metrics_by_borough.index, metrics_by_borough["RMSE (in 1000s)"], color='skyblue')
    #axes[0].set_title("RMSE by Borough")
    #axes[0].set_xlabel("Borough")
    #axes[0].set_ylabel("RMSE (in 1000s)")
    #axes[0].tick_params(axis='x', rotation=90)

    #axes[1].bar(metrics_by_borough.index, metrics_by_borough["MAE (in 1000s)"], color='orangered')
    #axes[1].set_title("MAE by Borough")
    #axes[1].set_xlabel("Borough")
    #axes[1].set_ylabel("MAE (in 1000s)")
    #axes[1].tick_params(axis='x', rotation=90)

    #plt.tight_layout()
    #st.pyplot(fig)

        # Scatter plot: Actual vs Predicted Average Price
    st.write("### Actual vs Predicted Average Price")
    st.warning("Actual vs. Predicted Plot: This scatter plot compares the actual AveragePrice with the predicted yhat (the 'prophet' ML model's price prediction). The red dashed line represents perfect predictions (where actual values equal predicted values). Points that are far from this line indicate larger prediction errors.")
    plt.figure(figsize=(10, 6))
    plt.scatter(cleaned_df['AveragePrice'], cleaned_df['yhat'], alpha=0.5, color='blue')
    plt.plot([cleaned_df['AveragePrice'].min(), cleaned_df['AveragePrice'].max()],
             [cleaned_df['AveragePrice'].min(), cleaned_df['AveragePrice'].max()],
             'r--', lw=2)
    plt.xlabel('Actual AveragePrice')
    plt.ylabel('Predicted yhat')
    plt.title('Actual vs Predicted Average Price')
    plt.grid(True)
    st.pyplot(plt)

    # Residual Analysis: Plotting Residuals (Actual - Predicted)
    st.write("### Residual Analysis")
    st.warning("Residual Analysis: The histogram of residuals shows the distribution of errors (actual minus predicted). Ideally, residuals should be centered around zero which would indicate that the model's errors are mostly small and are equally likely to be positive or negative. There should also be no significant skewness/kurtosis or patterns and when there is none present, it means the model lacks systematic bias or extreme errors in its predictions.")
    residuals = cleaned_df['AveragePrice'] - cleaned_df['yhat']
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residual Analysis')
    plt.grid(True)
    st.pyplot(plt)
    

# Section 4: Buy-To-Let Mortgage Calculator
def show_buy_to_let_mortgage_calculator():
    st.title("London Property Investment: Buy-To-Let :pound::handshake::house:")
    st.subheader("Mortgage Calculator :bank:")
    st.warning("Note: The interest rate is fixed in the calculation but in the Uk, the interest rate is fixed for a set period of time typically 2 to 10 years. Therefore when reading the repayment value, take into account that this only shows what the investor would be repaying during the fixed period.")

    def calculate_mortgage_payment(loan_amount, annual_interest_rate, loan_term_years):
        monthly_interest_rate = annual_interest_rate / 100 / 12
        number_of_payments = loan_term_years * 12
        mortgage_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments) / ((1 + monthly_interest_rate)**number_of_payments - 1)
        return mortgage_payment

    def calculate_rental_yield(annual_rental_income, property_price):
        rental_yield = (annual_rental_income / property_price) * 100
        return rental_yield

    def buy_to_let_calculator(property_price, deposit_percent, annual_interest_rate, loan_term_years, monthly_rental_income):
        deposit = property_price * (deposit_percent / 100)
        loan_amount = property_price - deposit
        monthly_mortgage_payment = calculate_mortgage_payment(loan_amount, annual_interest_rate, loan_term_years)
        
        annual_rental_income = monthly_rental_income * 12
        rental_yield = calculate_rental_yield(annual_rental_income, property_price)
        
        annual_mortgage_payment = monthly_mortgage_payment * 12
        net_profit = annual_rental_income - annual_mortgage_payment
        
        return {
            "Property Price": property_price,
            "Deposit": deposit,
            "Loan Amount": loan_amount,
            "Monthly Mortgage Payment": monthly_mortgage_payment,
            "Annual Rental Income": annual_rental_income,
            "Rental Yield": rental_yield,
            "Annual Mortgage Payment": annual_mortgage_payment,
            "Net Profit/Loss per Year": net_profit
        }

    col1, col2 = st.columns(2)
    property_price = col1.number_input("Property Price", min_value=5000, value=300000, step=5000)
    deposit_percent = col1.number_input("Deposit (%)", min_value=0, max_value=100, value=25, step=1)
    annual_interest_rate = col1.number_input("Annual Interest Rate (%)", min_value=0.0, value=4.5, step=0.05)
    loan_term_years = col2.number_input("Loan Term (Years)", min_value=1, max_value=40, value=25, step=1)
    monthly_rental_income = col2.number_input("Monthly Rental Income", min_value=0, value=1500, step=25)

    if st.button("Calculate"):
        st.session_state['result'] = buy_to_let_calculator(property_price, deposit_percent, annual_interest_rate, loan_term_years, monthly_rental_income)

    if 'result' in st.session_state:
        result = st.session_state['result']
        
        st.subheader("Results")
        
        col1, col2 = st.columns(2)

        col1.subheader("Mortgage Details")
        col2.subheader("Income & Profit Details")

        col1.metric("Property Price", f"£{result['Property Price']:,}")
        col1.metric("Deposit", f"£{result['Deposit']:,}")
        col1.metric("Loan Amount", f"£{result['Loan Amount']:,}")
        col1.metric("Monthly Mortgage Payment", f"£{result['Monthly Mortgage Payment']:.2f}")
        col1.metric("Annual Mortgage Payment", f"£{result['Annual Mortgage Payment']:.2f}")
        
        rental_yield_delta = f"{result['Rental Yield']:.2f}%" if result['Rental Yield'] >= 0 else f"{result['Rental Yield']:.2f}%"
        net_profit_delta = f"{result['Net Profit/Loss per Year']:.2f}" if result['Net Profit/Loss per Year'] >= 0 else f"{result['Net Profit/Loss per Year']:.2f}"
        
        col2.metric("Monthly Rental Income", f"£{result['Annual Rental Income']/12:,}")
        col2.metric("Annual Rental Income", f"£{result['Annual Rental Income']:,}")
        col2.metric("Rental Yield", f"{result['Rental Yield']:.2f}%", delta=rental_yield_delta)
        col2.metric("Net Profit/Loss per Year", f"£{result['Net Profit/Loss per Year']:.2f}", delta=net_profit_delta)

# Section 5: Average Rent
def show_average_rent():
    st.title("Average Rent across the London Boroughs in 2024")
    st.write("Courtesy of Zoopla, these are the current 2024 average rent prices by borough up to 2024 Q2.")
    
    
    data = pd.read_csv('data//MonthlyRentLondon.csv')
    data_clean = data[['Borough', 'Average Monthly Rent(£)', r'% change in the last 12 months']]
    st.dataframe(data_clean)
    
    # GEOSPATIAL MAP OF LONDON
    rent_data = pd.read_csv('data//MonthlyRentLondon.csv')

    # corrections
    name_corrections = {
        'City of Westminster': 'Westminster',
        'Barking and Dagenham': 'Barking & Dagenham',
        'Hammersmith and Fulham': 'Hammersmith & Fulham',
        'Kensington and Chelsea': 'Kensington & Chelsea',
        'Kingston upon Thames': 'Kingston upon Thames',
        'Richmond upon Thames': 'Richmond upon Thames',
        'W&sworth': "Wandsworth"
    }

    rent_data['Borough'] = rent_data['Borough'].str.replace('and', '&')
    rent_data['Borough'] = rent_data['Borough'].replace(name_corrections)

    geojson_path = 'data//London_Boroughs.geojson'
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    folium.Choropleth(
        geo_data=geojson_data,
        name='choropleth',
        data=rent_data,
        columns=['Borough', 'Average Monthly Rent(£)'],
        key_on='feature.properties.BOROUGH',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Monthly Rent (£)',
    ).add_to(m)

    #removes the need to click for smoother experience
    folium.GeoJson(
    geojson_data,
    name="Borough Labels",
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
    tooltip=folium.GeoJsonTooltip(fields=['BOROUGH'], aliases=['Borough:'])
    ).add_to(m)

    
    st.title("London Boroughs: Average Monthly Rent")
    folium_static(m)

# Section 6: RightMove Web Scraper
def fetch_rightmove_data(location, max_pages=5):
    base_url = f"https://www.rightmove.co.uk/property-to-rent/{location}.html"
    rent_values = []
    addresses = []
    property_types = []
    bedrooms = []

    for page in range(0, max_pages * 24, 24):  # RightMove uses increments of 24 for pagination
        url = f"{base_url}?index={page}"
        response = requests.get(url)
        if response.status_code != 200:
            st.error("Failed to retrieve data. Please check the location or try again later.")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        cards = soup.find_all('div', class_='l-searchResult')

        if not cards:
            break  # If no more results, break the loop

        for card in cards:
            # Rent
            rent = card.find('div', class_='propertyCard-rentalPrice-primary')
            rent_values.append(rent.text.strip() if rent else None)

            # Address
            address = card.find('address', class_='propertyCard-address')
            addresses.append(address.get('title') if address else None)

            # Property Type
            property_info = card.find('div', class_='property-information')
            if property_info:
                property_type = property_info.find('span', class_='propertyCard-title').text
                property_types.append(property_type.strip() if property_type else None)
            else:
                property_types.append(None)

            # Bedrooms
            bedroom_info = card.find('span', class_='propertyCard-title')
            if bedroom_info:
                # Extracting bedroom count from text, typically found in the title or description
                bedroom_match = re.search(r'(\d+)\s*bedroom', bedroom_info.text, re.IGNORECASE)
                bedrooms.append(bedroom_match.group(1) if bedroom_match else "N/A")
            else:
                bedrooms.append(None)

    # Ensure all lists have the same length
    max_length = max(len(rent_values), len(addresses), len(property_types), len(bedrooms))

    # If any list is shorter, append None values to make them the same length
    rent_values += [None] * (max_length - len(rent_values))
    addresses += [None] * (max_length - len(addresses))
    property_types += [None] * (max_length - len(property_types))
    bedrooms += [None] * (max_length - len(bedrooms))

    data = pd.DataFrame({
        'Rent': rent_values,
        'Address': addresses,
        'Property Type': property_types,
        'Bedrooms': bedrooms
    })
    
    return data

# Function to display RightMove data in Streamlit
def show_rightmove_web_scraper():
    st.title("RightMove **LIVE** Properties up for rent :mag_right:")

    location_input = st.text_input("Enter the London Borough you want to search (may take some time):")

    if st.button("Search"):
        data = fetch_rightmove_data(location_input)
        if data is not None:
            st.dataframe(data)

# Content will be displayed depending on which section the user selects
if selected_section == "Property Prices & Sales Volume":
    show_property_prices_sales_volume()
elif selected_section == "House Price Predictions":
    show_house_price_predictions()
elif selected_section == "Forecast Metrics":
    show_forecast_metrics()
elif selected_section == "Buy-To-Let Mortgage Calculator":
    show_buy_to_let_mortgage_calculator()
elif selected_section == "Average Rent & Map":
    show_average_rent()
elif selected_section == "RightMove **LIVE** Rents Up for Sale":
    show_rightmove_web_scraper()
elif selected_section == "Home":
    # Show all sections sequentially
    show_property_prices_sales_volume()
    st.divider()
    show_house_price_predictions()
    st.divider()
    show_forecast_metrics()
    st.divider()
    show_buy_to_let_mortgage_calculator()
    st.divider()
    show_average_rent()
    st.divider()
    show_rightmove_web_scraper()
#WEB FEATURES SECTION (MAYBE)
