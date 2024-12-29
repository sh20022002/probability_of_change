from datetime import datetime, timedelta
import pytz
import yfinance as yf
import os
import pandas as pd

SYMBOLS = ['NVDA', 'SPY', 'QQQ', 'IWM', 'TSLA']
CSV_PATH = r'C:\Users\shmue\projects\python\open_pojects\probability_of_market_movment\csvs'
TIMEFRAME = '1m'            # 1-minute data
DURATION_DAYS = 30           # How many days of data to fetch each run

def main():
    # Fetch data for all symbols and append to their respective CSV files
    get_data(SYMBOLS)
    # print(import_data(SYMBOLS[1], TIMEFRAME, DAYS=3))

def get_data(symbols):
    """
    For each symbol, read its CSV (if exists), figure out where we left off,
    fetch the next 3 days of 1m data from Yahoo Finance, and append to CSV.
    """
    for symbol in symbols:
        csv_file = os.path.join(CSV_PATH, f"{symbol}.csv")

        # 1) Try to read existing CSV to find the last Datetime
        if os.path.exists(csv_file):
            # We'll read the CSV columns we want: Datetime plus OHLCV
            try:
                df_existing = pd.read_csv(
                    csv_file,
                    parse_dates=['Datetime'],
                    index_col='Datetime'
                )
            except pd.errors.EmptyDataError:
                # If the file is empty, create an empty DataFrame
                df_existing = pd.DataFrame()

            if not df_existing.empty:
                last_date = df_existing.index.max()
                start_date = last_date + timedelta(minutes=1)
            else:
                # CSV might have a header but no rows
                start_date = datetime.now(tz=pytz.timezone('US/Eastern')) - timedelta(days=DURATION_DAYS)
        else:
            # No CSV => fetch the last 3 days from now
            start_date = datetime.now(tz=pytz.timezone('US/Eastern')) - timedelta(days=DURATION_DAYS)

        # 2) End date is 3 days after start_date
        end_date = start_date + timedelta(days=DURATION_DAYS)
        print(f"\nFetching data for {symbol} from {start_date} to {end_date}...")

        # 3) Fetch new data from yfinance
        df_new = import_data(symbol, TIMEFRAME, start_date, end_date)
        if df_new.empty:
            print(f"No new data returned for {symbol}")
            continue

        # 4) Filter out any rows that already exist
        if os.path.exists(csv_file) and not df_existing.empty:
            max_existing_index = df_existing.index.max()
            df_new = df_new.loc[df_new.index > max_existing_index]

        if df_new.empty:
            print(f"No new rows to append for {symbol}")
            continue

        
        # Here we rely on df_new **already** having ['Open','High','Low','Close','Volume']
        # plus a Datetime index. We'll specify index_label='Datetime' so it shows up in the CSV.
        df_new.to_csv(
            csv_file,
            mode='a',                     # Append mode          
            index_label=['Datetime']        # The index gets a column name "Datetime" in CSV
        )

        print(f"Appended {len(df_new)} new rows for {symbol} --> {csv_file}")

def import_data(symbol, timeframe, DAYS=3, start_date=None, end_date=None):

    stock_ticker = yf.Ticker(symbol)
    
    if not start_date and not end_date:

        end_date = datetime.now(tz=pytz.timezone('US/Eastern'))
        start_date = end_date - timedelta(days=DAYS)

        df = stock_ticker.history(start=start_date, end=end_date, interval=timeframe)
    
    else:
        df = stock_ticker.history(period='max', interval=timeframe)
        

    # Ensure the DataFrame is not empty
    if df.empty:
        raise(f"No data fetched for {stock}.")
        

    # Rename index if necessary
    if df.index.name != 'Datetime':
        df.index.name = 'Datetime'

    # Drop unnecessary columns if they exist
    columns_to_drop = [col for col in ['Dividends', 'Stock Splits'] if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise(f"Missing columns in data: {missing_columns}")
        

    # Select required columns
    df = df[required_columns]

    return df

if __name__ == '__main__':
    main()
