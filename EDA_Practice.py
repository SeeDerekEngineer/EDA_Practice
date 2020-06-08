#Obtained from https://medium.com/swlh/a-complete-guide-to-exploratory-data-analysis-and-data-cleaning-dd282925320f

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train = pd.read_csv("train.csv")

print(train.head())

train.rename(columns={'Id': 'id', 'PID': 'pid', 'MS SubClass': 'ms_subclass',}, inplace=True)

print(train.columns)

train.columns = [i.replace(' ', '_').lower() for i in train.columns]

print(train.columns)

def ames_eda(df):
    eda_df = {}
    eda_df['null_sum'] = df.isnull().sum()
    eda_df['null_pct'] = df.isnull().mean()
    eda_df['dtypes'] = df.dtypes
    eda_df['count'] = df.count()
    eda_df['mean'] = df.mean()
    eda_df['median'] = df.median()
    eda_df['std'] = df.std()
    eda_df['min'] = df.min()
    eda_df['max'] = df.max()

    return pd.DataFrame(eda_df)


summaryStatistics = ames_eda(train)

print(summaryStatistics)

trainWithExpandedStreet = pd.get_dummies(train, columns=["street"])  #Turns categories into one-hots.
                                                                     #cat.codes could have been used for label encoding

print(trainWithExpandedStreet.head())

correlations = train.corrwith(train['saleprice']).iloc[:-1].to_frame()
correlations['abs'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('abs', ascending=False)[0]
fig, ax = plt.subplots(figsize=(10,20))
sns.heatmap(sorted_correlations.to_frame(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax)

plt.show()

sns.boxplot(train['centralair'],
        train['saleprice']).set_title('Central Air vs. Sale Price')

plt.show()

sns.boxplot(train['kitchenqual'],
            train['saleprice']).set_title('Kitchen Quality vs. Sale Price')

plt.show()