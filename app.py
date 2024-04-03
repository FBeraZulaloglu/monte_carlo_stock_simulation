import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
        page_title="KeenSight - Monte Carlo Stock Simulation",
        page_icon=":chart_with_upwards_trend:",
    )
st.set_option('deprecation.showPyplotGlobalUse', False)
# Function to get S&P 500 stock symbols
def get_sp500_symbols():
    symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    return symbols

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(stock_data, num_simulations, num_days):
    returns = stock_data['Adj Close'].pct_change()
    mean_return = returns.mean()
    std_dev = returns.std()

    simulation_df = pd.DataFrame()

    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, std_dev, num_days)
        price_path = stock_data['Adj Close'].iloc[-1] * (1 + daily_returns).cumprod()
        simulation_df[f'Simulation {i+1}'] = price_path

    return simulation_df

# Set seed for reproducibility
np.random.seed(42)

st.image("logo.png", width=200)
# Streamlit app
st.title('Monte Carlo Stock Price Simulator')

# Get S&P 500 stock symbols
sp500_symbols = get_sp500_symbols()

# User input
symbol = st.selectbox('Select a stock symbol:', sp500_symbols, index=0)
start_date = st.date_input('Select a start date for historical data:', pd.to_datetime('2021-01-01'))

# Fetch stock data
stock_data = yf.download(symbol, start=start_date)

# Calculate the number of trading days in the next 3 months
num_days_3_months = int(np.busday_count(stock_data.index[-1].date(), (stock_data.index[-1] + pd.DateOffset(months=3)).date()))

# Perform Monte Carlo simulation
simulation_results = monte_carlo_simulation(stock_data, num_simulations=10, num_days=num_days_3_months)

# Display historical stock data
st.subheader('Historical Stock Prices')
st.line_chart(stock_data['Adj Close'])

# Display simulation results
st.subheader('Monte Carlo Simulation Results')
st.line_chart(simulation_results)

# Display statistical summary of simulations
st.subheader('Simulation Statistics')
st.write(simulation_results.describe())

# Additional summary statistics and visualizations

# Final Price Statistics
st.subheader('Final Price Statistics')
st.write(simulation_results.iloc[-1].describe())

# Risk-Return Metrics
sharpe_ratios = simulation_results.iloc[-1].pct_change().mean() / simulation_results.iloc[-1].pct_change().std()
st.subheader('Risk-Return Metrics')
st.write('Mean Sharpe Ratio:', sharpe_ratios.mean())
st.write('Standard Deviation of Final Prices:', simulation_results.iloc[-1].std())

# Histogram of Final Prices
st.subheader('Histogram of Final Prices')
plt.hist(simulation_results.iloc[-1], bins=20, edgecolor='black')
st.pyplot()

# Distribution of Percent Changes
st.subheader('Distribution of Percent Changes')
plt.hist(simulation_results.pct_change().iloc[-1], bins=20, edgecolor='black')
st.pyplot()

# Value at Risk (VaR)
confidence_level = 0.95  # Adjust as needed
var = simulation_results.iloc[-1].quantile(1 - confidence_level)
st.subheader(f'Value at Risk (VaR) at {confidence_level * 100}% Confidence Level')
st.write(var)


# Button to download simulation results as CSV
if st.button('Download Simulation Results as CSV'):
    # Save simulation results to CSV file in-memory
    csv_file = BytesIO()
    simulation_results.to_csv(csv_file, index=False)
    csv_file.seek(0)

    # Provide download button for the user
    st.download_button(
        label='Download CSV',
        data=csv_file,
        file_name=f'{symbol}_monte_carlo_simulation.csv',
        mime='text/csv',
        key=f'{symbol}_monte_carlo_simulation.csv',
        help='Click to download the CSV file.'
    )
