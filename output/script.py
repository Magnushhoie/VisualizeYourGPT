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

import matplotlib.pyplot as plt
import seaborn as sns

# Filter the dataframe to include only individuals with diabetes
df_diabetic = df_proc[df_proc['Outcome'] == 1]

# Create a scatter plot of insulin levels and BMI
plt.figure(figsize=(8, 6))
sns.scatterplot(x='BMI', y='Insulin', data=df_diabetic)
plt.title('Relationship between Insulin Levels and BMI in Individuals with Diabetes')
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.show()

# Create a residual plot of the linear regression model
plt.figure(figsize=(8, 6))
sns.residplot(x='BMI', y='Insulin', data=df_diabetic, lowess=True)
plt.title('Residual Plot of the Linear Regression Model of Insulin Levels on BMI in Individuals with Diabetes')
plt.xlabel('BMI')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.show()
