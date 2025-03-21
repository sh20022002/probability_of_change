import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score
from scipy.stats import linregress

SECTOR_TICKERS = {
    "S&P 500":      "SPY",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":        "XLE",
    "Financials":    "XLF",
    "Healthcare":    "XLV",
    "Industrials":   "XLI",
    "Materials":     "XLB",
    "Technology":    "XLK",
    "Telecom":       "VOX",
    "Utilities":     "XLU",
    "Oil":           "USO",
    "Gold":          "GLD",
    "Dollar":        "UUP",
    "Long Bond":     "TLT"
}

def analyze_stock_relationships(tickers, benchmark_tickers):
    """
    Analyzes stock relationships with benchmark tickers (SPY, RSP, VIX).

    Args:
        tickers (list): List of stock tickers.
        benchmark_tickers (list): List of benchmark tickers (e.g., ['SPY', 'RSP', 'VIX']).
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).

    Returns:
        dict: A dictionary containing correlation and linearity information.
    """

    try:
        all_tickers = tickers + benchmark_tickers
        data = yf.download(all_tickers, period='max')['Close']

        if data.empty:
            return {"error": "No data downloaded for the specified tickers and dates."}

        returns = data.pct_change().dropna()

        results = {}
        for ticker in tickers:
            results[ticker] = {}
            for benchmark in benchmark_tickers:

                # skips chacking covariance of itself
                if ticker == benchmark:
                    continue
                try:
                    returns_ticker = returns[ticker].values
                    returns_benchmark = returns[benchmark].values

                    correlation = np.corrcoef(returns_ticker, returns_benchmark)[0, 1]
                    slope, intercept, r_value, p_value, std_err = linregress(returns_ticker, returns_benchmark)
                    r_squared = r_value**2

                    results[ticker][benchmark] = {
                        "correlation": correlation,
                        "r_squared": r_squared,
                        "slope": slope,
                        "intercept": intercept,
                        "p_value": p_value,
                        "std_err": std_err,
                        "correlation_type": "positive" if correlation > 0.3 else ("negative" if correlation < -0.3 else "none")
                    }

                except Exception as e:
                    results[ticker][benchmark] = {"error": str(e)}

        return results

    except Exception as e:
        return {"error": str(e)}


def sector_covariance(sector_a: str, sector_b: str, lookback_days: int = 180) -> pd.DataFrame:
    """
    Returns long‑term covariance and covariance over the last `lookback_days`.
    
    Parameters:
      sector_a (str): Name of first sector (must match key in SECTOR_TICKERS)
      sector_b (str): Name of second sector
      lookback_days (int): Rolling window length in trading days (default=180)
    
    Returns:
      pd.DataFrame: Two rows — 'Long Term' & 'Last {lookback_days} days', 
                    one column 'Covariance'
    """
    try:
        ticker_a = SECTOR_TICKERS[sector_a]
        ticker_b = SECTOR_TICKERS[sector_b]
    except KeyError as e:
        raise ValueError(f"Unknown sector: {e.args[0]}. Available: {list(SECTOR_TICKERS)}")
    
    
    df = yf.download([ticker_a, ticker_b], period="max", progress=False)["Close"].dropna()
    returns = df.pct_change().dropna()
    
    
    long_cov = returns.cov().loc[ticker_a, ticker_b]
    recent_cov = returns.tail(lookback_days).cov().loc[ticker_a, ticker_b]
    
    return long_cov, recent_cov ,lookback_days

    
# ------------------------------------------------------------
tickers = ['SEDG']

benchmark_tickers = ['SPY', 'BTC', 'RSP', 'JPM', 'LLY']


results = analyze_stock_relationships(tickers, benchmark_tickers)

if "error" in results:
    print(f"Error: {results['error']}")
else:
    for ticker, benchmarks in results.items():
        print(f"\n{ticker} Relationships:")
        for benchmark, metrics in benchmarks.items():
            if "error" in metrics:
                print(f"  {benchmark}: Error: {metrics['error']}")
            else:
                print(f"  {benchmark}: Correlation = {metrics['correlation']:.4f}, Correlation Type: {metrics['correlation_type']}, R-squared = {metrics['r_squared']:.4f}, Slope = {metrics['slope']:.4f}")