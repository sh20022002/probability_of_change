import pandas as pd
import numpy as np
import polars as pl
from main import import_data 


def find_pattern(timeframe):
    hour = pl.Series()