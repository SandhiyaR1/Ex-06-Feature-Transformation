# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

# PROGRAM:

### Developed by: SANDHIYA R
### Register No. : 212222230129
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
df.head()

df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
# OUTPUT
### DATA
![image](https://user-images.githubusercontent.com/113497571/236632703-3ef0416d-ebca-43cf-b9dc-38905856e854.png)
### isnull().sum()
![image](https://user-images.githubusercontent.com/113497571/236632771-d18240dd-30ce-49ca-ac98-50bb41927129.png)
### df.info
![image](https://user-images.githubusercontent.com/113497571/236632795-d99f57e5-7f43-4a70-8c18-b09f2db889cd.png)
### df.discribe()
![image](https://user-images.githubusercontent.com/113497571/236632815-d6f17a5b-bc76-4984-9c3e-5833d264c14f.png)
### BEFORE TRANSFORMATION
![image](https://user-images.githubusercontent.com/113497571/236632842-d7801a82-76d5-4563-ac69-fcaf3e768e76.png)

![image](https://user-images.githubusercontent.com/113497571/236632847-9b7e828f-7360-4f9a-8097-5e821412d25b.png)
![image](https://user-images.githubusercontent.com/113497571/236632853-4ee1cb00-3548-4ccd-9ec8-a081c83ea587.png)

![image](https://user-images.githubusercontent.com/113497571/236632863-b3cc38ae-91ee-42d9-b938-3ab866c6a248.png)
### LOG TRANSFORMATION
![image](https://user-images.githubusercontent.com/113497571/236632884-4c9273d9-f19f-4ed6-8f4b-206db0b55f44.png)
### RECIPROCAL TRANSFORMATION
![image](https://user-images.githubusercontent.com/113497571/236632901-3c2049d6-e11f-459d-bb33-4f79e523db65.png)
### SQAURE RROT TRANSFORMATION
![image](https://user-images.githubusercontent.com/113497571/236632925-651760e0-3bb0-4d11-9ea6-c242abe72787.png)
![image](https://user-images.githubusercontent.com/113497571/236632929-19fa0474-9005-46ef-acba-df5146fce38c.png)
### POWER TRANSFORMATION
![image](https://user-images.githubusercontent.com/113497571/236632940-59079ffe-e36c-49f5-b0cb-47ec5bc6c7d4.png)
### QUANTILE TRANSFORMATION
# RESULT:
Thus feature transformation is done for the given dataset.
