import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

print('=' * 100)
print('[INFO] Reading CSV...')
df = pd.read_csv('./datasets/survey.csv')

print('=' * 100)
print('[INFO] Total Null Values Per Column:')
print(df.isna().sum())
print('Shape =', df.shape)

print('=' * 100)
print('[INFO] Percentage of Null Values in Dataset:')
nullPercentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
print('Percentage =', round(nullPercentage, 2), '%')

print('=' * 100)
print('[INFO] Dropping "state" column...')
df.drop(columns=['state'], inplace=True)

print('=' * 100)
print('[INFO] Filling missing values in "self_employed" column with mode...')
modeVal = df['self_employed'].mode()[0]
print('Mode =', modeVal)
df.loc[:, 'self_employed'] = df['self_employed'].fillna(modeVal)

print('=' * 100)
print('[INFO] Applying SimpleImputer on "work_interfere" with most_frequent strategy...')
print('Column DataTypes:\n', df.dtypes)
print('Integer Columns:', df.select_dtypes(include='int').columns.tolist())

si = SimpleImputer(strategy='most_frequent')
df['work_interfere'] = si.fit_transform(df[['work_interfere']]).ravel()

print('[INFO] Imputation complete.')
print('=' * 100)

ohe = OneHotEncoder(drop='first')
enData = ['self_employed', 'family_history', 'treatment', 'work_interfere', 'remote_work',
'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
'mental_vs_physical', 'obs_consequence']

encoded = ohe.fit_transform(df[enData])
encodedDf = pd.DataFrame(encoded.toarray(), columns=ohe.get_feature_names_out(enData), index=df.index)

df.drop(columns=enData, inplace=True)
df = pd.concat([df, encodedDf], axis=1)

print('[INFO] One hot encoding completed.')
print('=' * 100)

print('df \n' , encodedDf)