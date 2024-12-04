from datetime import datetime, timedelta
import yfinance as yf
import polars as pl
import pytz
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pandas as pd




STOCK = 'BITCOIN-USD'
SYMBOL = 'NVDA'
DAYS = 366
timeframe = '1d'


def main():
    global STOCK
    df = import_data(timeframe)
    

def import_data(timeframe):
    global SYMBOL, DAYS

    end_date = get_exchange_time()
    start_date = end_date - timedelta(days=DAYS)

    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(start=start_date, end=end_date, interval=timeframe)
    # Ensure the DataFrame is not empty
    if df.empty:
        raise ValueError("The dataframe is empty.")

    # Rename index if necessary
    if df.index.name != 'Datetime':
        df.index.name = 'Datetime'

    # Drop unnecessary columns if they exist
    columns_to_drop = [col for col in ['Dividends', 'Stock Splits', 'Capital Gains'] if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    df = pl.from_pandas(df, include_index=True)
    return df

def analyze(df):
    '''takes a data frame(df) and takes open and close of every columm and calculates the change and adds up and calculates the probability 
    of change for each timeframe
    
    calls import data for each timeframe:
     - q (q1, q2 ...q4)
     - month (m1...m12)
     - week (w1 ...w54)
     - day (m, t_3, w, t_5, f)
     - hour (h1...h8) in trading hours 
     '''

def plot_res(df, ticker):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker))
    return fig

def plot_volume(df, ticker):
    
    fig = px.line(x=df['Datetime'], y=df['Volume'])
    return fig

def patterns(df):
    # Convert 'Close' prices and 'Datetime' to numpy arrays
    close_prices = df['Close'].to_numpy()
    datetime_values = df['Datetime'].to_numpy()

    # Find local maxima
    local_max = argrelextrema(close_prices, np.greater)[0]

    # Define tolerance as a percentage of price
    tolerance = 0.02  # 2% tolerance

    patterns = []

    for i in range(1, len(local_max) - 1):
        left_shoulder = local_max[i - 1]
        head = local_max[i]
        right_shoulder = local_max[i + 1]

        # Calculate relative difference between shoulders
        avg_shoulder_price = (close_prices[left_shoulder] + close_prices[right_shoulder]) / 2
        shoulder_diff = abs(close_prices[left_shoulder] - close_prices[right_shoulder]) / avg_shoulder_price

        if (close_prices[left_shoulder] < close_prices[head] and
            close_prices[right_shoulder] < close_prices[head] and
            shoulder_diff < tolerance):
            patterns.append((left_shoulder, head, right_shoulder))

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(datetime_values, close_prices, label='Close Price')

    for pattern in patterns:
        ls, h, rs = pattern
        # Get the datetime values for the indices
        dates = [datetime_values[ls], datetime_values[h], datetime_values[rs]]
        prices = [close_prices[ls], close_prices[h], close_prices[rs]]
        plt.plot(dates, prices, 'ro')
        plt.plot(dates, prices, 'r--')

    plt.title('Detected Head and Shoulders Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def detect_cup_and_handle(df, window=5, tolerance=0.02, handle_max_retrace=0.5, target_ratio=0.95):
    """
    Detects Cup and Handle patterns in the given DataFrame.

    Parameters:
    - df: Polars DataFrame with 'Datetime' and 'Close' columns.
    - window: Integer, smoothing window size for cup detection.
    - tolerance: Float, minimum cup depth relative to price (e.g., 0.02 for 2%).
    - handle_max_retrace: Float, maximum retracement of the handle relative to cup depth (e.g., 0.5 for 50%).
    - target_ratio: Float, the percentage of the cup depth the price should rise after the handle to consider the pattern successful.

    Returns:
    - patterns: List of dictionaries containing pattern details.
    """
    # Convert 'Close' prices and 'Datetime' to numpy arrays
    close_prices = df['Close'].to_numpy()
    datetime_values = df['Datetime'].to_numpy()
    
    # Smooth the data to reduce noise for cup detection
    close_prices_smooth = np.convolve(close_prices, np.ones(window)/window, mode='valid')
    dates_smooth = datetime_values[window - 1:]
    
    # Find local minima and maxima in the smoothed data for cup detection
    local_min = argrelextrema(close_prices_smooth, np.less)[0]
    local_max = argrelextrema(close_prices_smooth, np.greater)[0]
    
    patterns = []
    
    # Loop through local minima to find potential cups
    for i in range(1, len(local_min)):
        cup_bottom_idx = local_min[i]
        
        # Find the left peak (local max before cup bottom)
        left_peaks = local_max[local_max < cup_bottom_idx]
        if len(left_peaks) == 0:
            continue
        left_peak_idx = left_peaks[-1]
        
        # Find the right peak (local max after cup bottom)
        right_peaks = local_max[local_max > cup_bottom_idx]
        if len(right_peaks) == 0:
            continue
        right_peak_idx = right_peaks[0]
        
        # Ensure the cup is symmetric within tolerance
        left_peak_price = close_prices_smooth[left_peak_idx]
        right_peak_price = close_prices_smooth[right_peak_idx]
        cup_bottom_price = close_prices_smooth[cup_bottom_idx]
        
        # Check if left and right peaks are approximately equal
        peak_diff = abs(left_peak_price - right_peak_price) / ((left_peak_price + right_peak_price) / 2)
        if peak_diff > tolerance:
            continue
        
        # Calculate cup depth
        cup_depth = ((left_peak_price + right_peak_price) / 2) - cup_bottom_price
        cup_height = (left_peak_price + right_peak_price) / 2
        
        # Ensure cup depth is significant relative to price
        if cup_depth / cup_height < tolerance:
            continue
        
        # Map indices back to original data for handle detection
        handle_start_idx_original = right_peak_idx + (window - 1)
        
        # Use original data for handle detection
        handle_end_idx_original = handle_start_idx_original + 1
        while (handle_end_idx_original < len(close_prices) and
               close_prices[handle_end_idx_original] <= close_prices[handle_end_idx_original - 1]):
            handle_end_idx_original += 1
        
        # If handle_end_idx_original reached the end of data, skip
        if handle_end_idx_original >= len(close_prices):
            continue
        
        # Handle retracement should not exceed max retracement
        handle_min_price = np.min(close_prices[handle_start_idx_original:handle_end_idx_original + 1])
        handle_retrace = (right_peak_price - handle_min_price) / cup_depth
        if handle_retrace > handle_max_retrace:
            continue
        
        # Check if price rises after handle completion (using original data)
        potential_target_price = right_peak_price + cup_depth * target_ratio
        post_handle_prices = close_prices[handle_end_idx_original:]
        target_reached = np.any(post_handle_prices >= potential_target_price)
        
        # Calculate potential upside in percent
        potential_upside = (potential_target_price - right_peak_price) / right_peak_price * 100
        
        # Record pattern details
        patterns.append({
            'cup_start_idx': left_peak_idx + (window - 1),
            'cup_bottom_idx': cup_bottom_idx + (window - 1),
            'cup_end_idx': right_peak_idx + (window - 1),
            'handle_start_idx': handle_start_idx_original,
            'handle_end_idx': handle_end_idx_original - 1,
            'potential_upside_%': potential_upside,
            'target_reached': target_reached,
            'cup_depth': cup_depth,
            'pattern_start_date': datetime_values[left_peak_idx + (window - 1)],
            'pattern_end_date': datetime_values[handle_end_idx_original - 1]
        })
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(datetime_values, close_prices, label='Close Price', alpha=0.5)
    plt.plot(dates_smooth, close_prices_smooth, label='Smoothed Close Price', color='blue')
    
    for pattern in patterns:
        # Get indices
        cup_start_idx = pattern['cup_start_idx']
        cup_bottom_idx = pattern['cup_bottom_idx']
        cup_end_idx = pattern['cup_end_idx']
        handle_start_idx = pattern['handle_start_idx']
        handle_end_idx = pattern['handle_end_idx']
        
        # Get dates and prices for cup (smoothed data)
        cup_dates = datetime_values[[cup_start_idx, cup_bottom_idx, cup_end_idx]]
        cup_prices = close_prices[[cup_start_idx, cup_bottom_idx, cup_end_idx]]
        
        # Get dates and prices for handle (original data)
        handle_dates = datetime_values[handle_start_idx:handle_end_idx+1]
        handle_prices = close_prices[handle_start_idx:handle_end_idx+1]
        
        # Plot cup
        plt.plot(cup_dates, cup_prices, 'ro-', label='Cup' if pattern == patterns[0] else "")
        # Plot handle
        plt.plot(handle_dates, handle_prices, 'go-', label='Handle' if pattern == patterns[0] else "")
        # Highlight the pattern
        plt.fill_between(datetime_values[cup_start_idx:handle_end_idx+1], close_prices[cup_start_idx:handle_end_idx+1], alpha=0.2)
        
        # Annotate potential upside
        plt.annotate(f"Potential Upside: {pattern['potential_upside_%']:.2f}%", (datetime_values[cup_end_idx], close_prices[cup_end_idx]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Detected Cup and Handle Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Return patterns for further analysis
    return patterns

def get_exchange_time():
    '''
    Get the current time in New York.

    Returns:
    - datetime: The current time in New York.
    '''
    ny_timezone = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_timezone)
    return ny_time

def analyze_daily_statistics(df, columns=['High', 'Close', 'Low', 'Volume']):
    # for 1m timframe
    # Ensure 'Datetime' is of datetime type
    df = df.with_columns(pl.col('Datetime').cast(pl.Datetime))
    
    # Extract 'Hour' from 'Datetime'
    df = df.with_columns(pl.col('Datetime').dt.hour().alias('Hour'))
    
    # Extract 'Date' from 'Datetime'
    df = df.with_columns(pl.col('Datetime').dt.date().alias('Date'))
    
    # Get the list of unique dates to compute probabilities
    dates = df.select('Date').unique()
    total_days = dates.height
    
    # Compute statistics per hour
    hourly_stats = {}
    for column in columns:
        # Group by 'Hour' and compute statistics
        stats = df.group_by('Hour').agg([
            pl.col(column).mean().alias('Mean'),
            pl.col(column).median().alias('Median'),
            pl.col(column).quantile(0.2).alias('20th Percentile'),
            pl.col(column).quantile(0.4).alias('40th Percentile'),
            pl.col(column).quantile(0.6).alias('60th Percentile'),
            pl.col(column).quantile(0.8).alias('80th Percentile'),
        ]).sort('Hour')
        
        hourly_stats[column] = stats.to_pandas()
    
    # Identify hours of max and min events for 'Volume' and 'Close'
    max_min_events = {}
    for column in ['Volume', 'Close']:
        # Find the max and min values per day
        daily_max_min = df.group_by('Date').agg([
            pl.col(column).max().alias('Daily Max'),
            pl.col(column).min().alias('Daily Min')
        ])
        
        # Join back to get the hours when max and min occur
        df_with_daily = df.join(daily_max_min, on='Date', how='left')
        
        # Filter rows where the value equals the daily max or min
        max_rows = df_with_daily.filter(pl.col(column) == pl.col('Daily Max'))
        min_rows = df_with_daily.filter(pl.col(column) == pl.col('Daily Min'))
        
        # Count occurrences per hour
        max_hour_counts = max_rows.group_by('Hour').count().select([
            'Hour',
            pl.col('count').alias('Max Count')
        ])
        min_hour_counts = min_rows.group_by('Hour').count().select([
            'Hour',
            pl.col('count').alias('Min Count')
        ])
        
        # Compute probabilities
        max_hour_counts = max_hour_counts.with_columns(
            (pl.col('Max Count') / total_days * 100).alias('Max Probability (%)')
        ).to_pandas()
        
        min_hour_counts = min_hour_counts.with_columns(
            (pl.col('Min Count') / total_days * 100).alias('Min Probability (%)')
        ).to_pandas()
        
        max_min_events[column] = {
            'Max': max_hour_counts,
            'Min': min_hour_counts,
        }
    
    # Plotting statistics per hour
    num_columns = len(columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(14, 4 * num_columns))
    
    if num_columns == 1:
        axes = [axes]
    
    for idx, column in enumerate(columns):
        ax = axes[idx]
        stats = hourly_stats[column]
        
        # Plot percentiles
        ax.fill_between(stats['Hour'], stats['20th Percentile'], stats['80th Percentile'], color='lightblue', alpha=0.5, label='20th-80th Percentile Range')
        ax.plot(stats['Hour'], stats['Mean'], label='Mean', color='blue')
        ax.plot(stats['Hour'], stats['Median'], label='Median', color='green')
        
        # Set labels and title
        ax.set_title(f'{column} Statistics Per Hour')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'{column} Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plotting max and min event probabilities for 'Volume' and 'Close'
    for column in ['Volume', 'Close']:
        fig, ax = plt.subplots(figsize=(14, 4))
        
        max_probs = max_min_events[column]['Max']
        min_probs = max_min_events[column]['Min']
        
        # Merge max and min probabilities
        probs = pd.merge(max_probs, min_probs, on='Hour', how='outer').fillna(0)
        probs = probs.sort_values('Hour')
        
        ax.bar(probs['Hour'] - 0.2, probs['Max Probability (%)'], width=0.4, label='Max', color='red', align='center')
        ax.bar(probs['Hour'] + 0.2, probs['Min Probability (%)'], width=0.4, label='Min', color='green', align='center')
        
        ax.set_title(f'Probability of Max and Min {column} Occurring at Each Hour')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Probability (%)')
        ax.legend()
        ax.grid(True)
        ax.set_xticks(range(0, 24))
        plt.tight_layout()
        plt.show()
        
        # Print the probabilities
        print(f"\nProbability of Max and Min {column} Occurring at Each Hour:")
        print(probs[['Hour', 'Max Probability (%)', 'Min Probability (%)']].to_string(index=False))

if __name__ == '__main__':
    main()