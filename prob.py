import pandas as pd

import numpy as np

import polars as pl

from main import import_data 



def find_gaps_and_return_open_gap(df: pd.DataFrame):

    """

    data: A DataFrame with at least the columns:

       - 'open'

       - 'close'

       - 'high'

       - 'low'

    Rows should be in chronological order (oldest to newest).


    This function will:

        1) Add 'prev_close' (yesterday's close),

        2) Add 'gap' (today's open - yesterday's close),

        3) Add 'gap_pct' = gap / prev_close * 100,

        4) Add 'gap_closed' (True/False) depending on whether today's

           price action revisited yesterday's close.

        5) Finally, it inspects the *last row* of the DataFrame:

           - If there is a non-zero gap and 'gap_closed' == False,

             return (gap, gap_pct).

           - Otherwise return None.


    Returns:

       - None OR (gap_price, gap_pct) if the gap remains open in the last row.

    """


    # 1. Shift to get 'prev_close'
    df['prev_close'] = df['Close'].shift(1)


    # 2. gap = today's open - yesterday's close
    df['gap'] = df['Open'].drop.index_col - df['prev_close'].drop.index_col


    # 3. gap_pct = gap / prev_close * 100
    df['gap_pct'] = (df['gap'] / df['prev_close']) * 100


    # 4. Determine if gap is up or down

    gap_up   = df['Open'] > df['prev_close']

    gap_down = df['Open'] < df['prev_close']


    # Mark gap_closed = True if price returned to yesterday's close
    df['gap_closed'] = False
    df.loc[gap_up   & (df['low']  <= df['prev_close']), 'gap_closed'] = True
    df.loc[gap_down & (df['high'] >= df['prev_close']), 'gap_closed'] = True


    # OPTIONAL: drop the first row where prev_close is NaN

    # data.dropna(subset=['prev_close'], inplace=True)


    # 5. Check the *last row* to see if the gap remained open

    last_row = df.iloc[-1]
    

    # If gap != 0 and it did not close, return (gap, gap_pct)

    if pd.notna(last_row['gap']) and last_row['gap'] != 0 and not last_row['gap_closed']:

        return (last_row['gap'], last_row['gap_pct'])

    else:

        return None, None