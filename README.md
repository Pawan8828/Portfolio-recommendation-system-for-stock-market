# Portfolio-recommendation-system-for-stock-market
Python |Pandas| NumPy |Seaborn| Streamlets | YFinance | K-Means | cvxpy | Data Visualization

Designed and deployed an intelligent system to help investors build optimized stock portfolios based on risk-return analysis and fundamental metrics.

• Fetched and analyzed historical stock data (prices, returns, PE ratio, EPS) using the yFinance API for informed decision-making.

• Clustered stocks with similar performance characteristics using K-Means clustering after preprocessing with StandardScaler.

• Formulated a quadratic programming model using cvxpy to optimize stock allocations under investment constraints (e.g., min 5%, max 40% per stock).

• Incorporated fundamental analysis by integrating valuation metrics like PE ratio and EPS into the stock selection logic.

• Developed an interactive Streamlit web app that allows users to: * Input custom stock tickers. * View optimized allocations and performance metrics. * Visualize results via charts and plots.

• Enforced dynamic allocation constraints to ensure realistic, diversified, and balanced portfolios aligned with investor preferences.

Outcome: Enabled users to simulate, evaluate, and personalize stock portfolios using a data-driven approach—combining historical trends, machine learning, and optimization techniques for smarter investing.
