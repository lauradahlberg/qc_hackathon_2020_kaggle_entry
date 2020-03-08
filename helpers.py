import pandas as pd
import numpy as np
from IPython.display import display



def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000, "display.max_colwidth", 1000):
        display(df)
        
        
def add_datepart(df, fldname, drop=True):
    """
    The add_datepart method extracts particular date fields from a complete datetime 
    for the purpose of constructing categoricals. 
    Expanding date-time into additional fields to capture any trend/cyclical behavior 
    as a function of time at any of these granularities.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)
        

# for when the data is time-based
# https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/machine-learning/2017-edition/lesson-2-random-forest-deep-dive.md
def split_vals(a, n): return a[:n].copy(), a[n:].copy()

# n_valid = 12000 # same as Kaggle's test set size
# n_trn = len(df) - n_valid

# raw_train, raw_valid = split_vals(df_raw, n_trn)
# X_train, X_valid = split_vals(df, n_trn)
# y_train, y_valid = split_vals(y, n_trn)
