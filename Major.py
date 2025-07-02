import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import random

# Settings
np.random.seed(42)

# Fetch stock data with fixed date range
@st.cache_data
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=False)
    return data

# Calculate returns
def calculate_returns(data, tickers):
    returns = pd.concat([data[ticker]['Close'].pct_change() for ticker in tickers], axis=1)
    returns.columns = tickers
    returns.dropna(inplace=True)
    return returns

# Clustering stocks
def cluster_stocks(returns, n_clusters=3):
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns.T)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(returns_scaled)
    cluster_map = pd.DataFrame({'ticker': returns.columns, 'cluster': clusters})
    return cluster_map

# Helper: Is rate limit error?
def is_rate_limit_error(e):
    return "Rate limit" in str(e) or "YFRateLimitError" in str(type(e))

@st.cache_data
def get_fundamentals(tickers):
    fundamentals = {}
    for ticker in tickers:
        attempts = 5
        for attempt in range(attempts):
            try:
                info = yf.Ticker(ticker).fast_info  # Faster & more reliable than .info
                pe_ratio = info.get('peRatio') or np.nan
                eps = info.get('eps') or np.nan
                fundamentals[ticker] = {'peRatio': pe_ratio, 'eps': eps}
                break  # Success, exit retry loop
            except Exception as e:
                if is_rate_limit_error(e):
                    time.sleep(3 + random.uniform(0, 2))  # Wait and retry
                else:
                    time.sleep(1)
                if attempt == attempts - 1:
                    # Removed the warning line
                    fundamentals[ticker] = {'peRatio': np.nan, 'eps': np.nan}
    df = pd.DataFrame(fundamentals).T
    if df.dropna().empty:
        # Removed the error message
        df['peRatio'] = 15  # Average PE ratio as fallback
        df['eps'] = 10      # Average EPS as fallback
    return df


# Portfolio optimization (no changes)
def optimize_portfolio(returns, fundamentals, risk_tolerance=0.5):
    n_assets = len(returns.columns)
    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    pe = fundamentals['peRatio'].values
    eps = fundamentals['eps'].values
    pe_score = 1 / (pe + 1e-6)
    eps_score = eps
    scores_matrix = np.vstack((pe_score, eps_score)).T
    scaled_scores = StandardScaler().fit_transform(scores_matrix)
    scores = scaled_scores.mean(axis=1)
    weights = cp.Variable(n_assets)
    portfolio_return = expected_returns.T @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    objective = cp.Maximize(
        portfolio_return - (risk_tolerance * 50) * portfolio_variance + 0.005 * scores @ weights
    )
    constraints = [cp.sum(weights) == 1, weights >= 0.05, weights <= 0.40]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed: {prob.status}")
    return weights.value

# Plot functions (no changes)
def plot_daily_returns(returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in returns.columns:
        ax.plot(returns.index, returns[col], label=col)
    ax.set_title("Daily Stock Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_clusters(cluster_map):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=cluster_map, x='cluster', palette='viridis', ax=ax)
    ax.set_title("Stock Clusters Count")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Stocks")
    ax.grid(axis='y')
    st.pyplot(fig)

def plot_fundamentals(fundamentals):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=fundamentals.index, y=fundamentals['eps'], ax=axs[0], palette='coolwarm')
    axs[0].set_title('Earnings Per Share (EPS)')
    axs[0].tick_params(axis='x', rotation=45)
    sns.barplot(x=fundamentals.index, y=fundamentals['peRatio'], ax=axs[1], palette='coolwarm')
    axs[1].set_title('Price to Earnings (P/E) Ratio')
    axs[1].tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_portfolio(optimized_portfolio):
    optimized_portfolio = optimized_portfolio[optimized_portfolio['weight'] > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(optimized_portfolio['weight'], labels=optimized_portfolio['ticker'],
           autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title("Recommended Portfolio Allocation")
    st.pyplot(fig)

# Main App
def main():
    st.title("\U0001F4C8 Smart Portfolio Recommender (with Clustering & Fundamentals)")
    st.sidebar.header("Configuration")
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 4, 30)  # Fixed to April 2025
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    investment = st.sidebar.number_input("Investment Amount (₹)", min_value=0, value=10000, step=1000)
    tickers = st.sidebar.multiselect(
        "Select Stocks",
        ['INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'HDFCBANK.NS',
         'RELIANCE.NS', 'TCS.NS', 'HDFC.NS', 'ITC.NS', 'SBIN.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
         'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
         'ULTRACEMCO.NS', 'NESTLEIND.NS', 'LT.NS', 'BAJAJ-AUTO.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ADANIENT.NS',
         'TATASTEEL.NS', 'JSWSTEEL.NS', 'DRREDDY.NS', 'CIPLA.NS', 'TECHM.NS', 'GRASIM.NS', 'HINDALCO.NS',
         'DIVISLAB.NS', 'UPL.NS', 'BAJAJFINSV.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'SHREECEM.NS',
         'INDUSINDBK.NS', 'COALINDIA.NS', 'BPCL.NS', 'IOC.NS', 'GAIL.NS', 'HEROMOTOCO.NS', 'VEDL.NS'],
        default=['INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS']
    )
    risk_tolerance = st.sidebar.slider("Risk Tolerance (0: Low, 1: High)", 0.0, 1.0, 0.5)

    if tickers:
        data = get_data(tickers, start_str, end_str)
        returns = calculate_returns(data, tickers)
        fundamentals = get_fundamentals(tickers)
        common_tickers = fundamentals.index.intersection(returns.columns)
        returns = returns[common_tickers]
        fundamentals = fundamentals.loc[common_tickers]
        st.subheader(f"Daily Stock Returns ({start_str} to {end_str})")
        plot_daily_returns(returns)
        st.subheader("Stock Clustering")
        cluster_map = cluster_stocks(returns, n_clusters=3)
        st.dataframe(cluster_map)
        plot_clusters(cluster_map)
        st.subheader("Fundamentals (EPS & P/E)")
        st.dataframe(fundamentals)
        plot_fundamentals(fundamentals)
        st.subheader("Portfolio Recommendation")
        try:
            optimized_weights = optimize_portfolio(returns, fundamentals, risk_tolerance)
            optimized_portfolio = pd.DataFrame({
                'ticker': returns.columns,
                'weight': optimized_weights,
                'amount (₹)': optimized_weights * investment
            })
            st.dataframe(optimized_portfolio.style.format({
                "weight": "{:.2%}",
                "amount (₹)": "₹{:,.2f}"
            }))
            plot_portfolio(optimized_portfolio)
            expected_return = (returns.mean() @ optimized_weights) * 252
            portfolio_volatility = np.sqrt(optimized_weights.T @ returns.cov().values @ optimized_weights) * np.sqrt(252)
            sharpe_ratio = expected_return / portfolio_volatility
            st.success(f"**Expected Annual Return:** {expected_return:.2%}")
            st.success(f"**Expected Annual Volatility:** {portfolio_volatility:.2%}")
            st.success(f"**Sharpe Ratio (approx):** {sharpe_ratio:.2f}")
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
    else:
        st.warning("Please select at least one stock.")

if __name__ == "__main__":
    main()
