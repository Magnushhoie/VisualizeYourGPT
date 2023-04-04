import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels
import scipy
import scipy.stats as stats
import sklearn

# Print everything
pd.option_context(
    'display.max_rows', 40,
    'display.max_columns', 40,
    'display.precision', 3,
    )

df_proc = pd.read_csv("data/data.csv")
