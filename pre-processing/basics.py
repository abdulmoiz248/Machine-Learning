import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns


print('=' * 100)
print('[INFO] Reading CSV...')
df = pd.read_csv('./datasets/survey.csv')



print('=' * 100)
print('[INFO] Dropping "state" column...')
df.drop(columns=['state'], inplace=True)


print('=' * 100)
print('[INFO] Adjusting age column')
print(df["Age"].unique())       # See weird values
print(df["Age"].dtype)    
df = df[df["Age"].between(10, 100)]


print('=' * 100)
print('[INFO] Filling missing values in "self_employed" column with mode...')
modeVal = df['self_employed'].mode()[0]
df.loc[:, 'self_employed'] = df['self_employed'].fillna(modeVal)

print('=' * 100)
print('[INFO] Imputing "work_interfere" with most frequent...')
si = SimpleImputer(strategy='most_frequent')
df['work_interfere'] = si.fit_transform(df[['work_interfere']]).ravel()

print('=' * 100)
print('[INFO] Label encoding ordinal columns...')



la=LabelEncoder()
df['work_interfere'] = la.fit_transform(df['work_interfere'])
df['leave']=la.fit_transform(df['leave'])
print('[INFO] Label encoding done.')

print('=' * 100)
print('[INFO] One hot encoding categorical columns...')

ohe = OneHotEncoder(drop='first', sparse_output=False)
enData = ['self_employed', 'family_history', 'treatment', 'remote_work',
'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
'anonymity', 'mental_health_consequence', 'phys_health_consequence',
'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
'mental_vs_physical', 'obs_consequence']

encoded = ohe.fit_transform(df[enData])
encodedDf = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(enData), index=df.index)

df.drop(columns=enData, inplace=True)
df = pd.concat([df, encodedDf], axis=1)

print('[INFO] Encoding complete.')
print('=' * 100)


print('[INFO] Dataset Description')
print(df.describe())
print('=' * 100)

sns.boxplot(x="Age",data=df)
plt.show()


