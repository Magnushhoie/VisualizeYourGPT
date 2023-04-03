import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

# Print everything
pd.option_context(
    'display.max_rows', 40,
    'display.max_columns', 40,
    'display.precision', 3,
    )

dataset = pd.read_csv("data/data.csv")

import pandas as pd
import numpy as np
import scipy.stats as stats

#Check for missing values
print("Missing values per column:\n", dataset.isna().sum())

#Print number of unique values for each column
print("Number of unique values per column:\n", dataset.nunique())

#Print correlation matrix
print("Correlation Matrix:\n", dataset.corr())

#Calculate and print p-value for correlation between Glucose and Outcome
glucose_corr, glucose_pvalue = stats.pearsonr(dataset.Glucose, dataset.Outcome)
print("Correlation between Glucose and Outcome p-value:", glucose_pvalue)

#Calculate and print mean glucose value for each Outcome category
print("Mean Glucose by Outcome:\n", dataset.groupby('Outcome').Glucose.mean())

#Calculate and print mean age value for each Outcome category
print("Mean Age by Outcome:\n", dataset.groupby('Outcome').Age.mean())

#Calculate and print proportion of diabetic patients
diabetic_proportion = dataset.Outcome.mean()
print("Proportion of diabetic patients:", diabetic_proportion)

#Calculate and print proportion of non-diabetic patients
non_diabetic_proportion = 1 - diabetic_proportion
print("Proportion of non-diabetic patients:", non_diabetic_proportion)

#Calculate and print median number of pregnancies for diabetic patients
print("Median number of pregnancies for diabetic patients:", dataset[dataset.Outcome == 1].Pregnancies.median())

#Calculate and print median number of pregnancies for non-diabetic patients
print("Median number of pregnancies for non-diabetic patients:", dataset[dataset.Outcome == 0].Pregnancies.median())

import pandas as pd
import numpy as np
import scipy.stats as stats

#Print quartiles for Glucose and Insulin columns
print("Glucose quartiles:\n", dataset.Glucose.quantile([0.25, 0.5, 0.75]))
print("Insulin quartiles:\n", dataset.Insulin.quantile([0.25, 0.5, 0.75]))

#Print number of patients with zero values for each feature
zero_counts = (dataset == 0).sum()
print("Number of zero values per feature:\n", zero_counts)

#Calculate and print proportion of patients with zero values for each feature
zero_proportions = zero_counts / len(dataset)
print("Proportion of zero values per feature:\n", zero_proportions)

#Print histograms for each feature
dataset.hist(figsize=(10, 10))

#Print boxplots for each feature
dataset.boxplot(figsize=(10, 6))


### More

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(dataset['Outcome'], dataset['Glucose'])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Glucose as a Predictor of Diabetes')
plt.legend(loc="lower right")
plt.show()


import seaborn as sns

corr = dataset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap for All Features')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=dataset)
plt.title('Scatter Plot of BMI and Glucose, Colored by Outcome')
plt.show()




from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(dataset['Age'], event_observed=dataset['Outcome'])
kmf.plot()
plt.title('Kaplan-Meier Survival Curve for Diabetes Onset as a Function of Age')
plt.show()



plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Pregnancies', data=dataset)
plt.title('Box Plot of Number of Pregnancies, Grouped by Outcome')
plt.show()




