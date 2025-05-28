import pandas as pd
from sklearn.impute import SimpleImputer

print('=' * 100)
print('[INFO] Reading CSV...')
df = pd.read_csv('./datasets/survey.csv')

print('=' * 100)
print('[INFO] Total Null Values Per Column:')
print(df.isna().sum())
print('Shape =', df.shape)

print('=' * 100)
print('[INFO] Percentage of Null Values in Dataset:')
null_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
print('Percentage =', round(null_percentage, 2), '%')

print('=' * 100)
print('[INFO] Dropping "state" column...')
df.drop(columns=['state'], inplace=True)

print('=' * 100)
print('[INFO] Filling missing values in "self_employed" column with mode...')
mode_val = df['self_employed'].mode()[0]
print('Mode =', mode_val)
df['self_employed'].fillna(mode_val, inplace=True)

print('=' * 100)
print('[INFO] Applying SimpleImputer on "work_interfere" with most_frequent strategy...')
print('Column DataTypes:\n', df.dtypes)
print('Integer Columns:', df.select_dtypes(include='int').columns.tolist())

si = SimpleImputer(strategy='most_frequent')
df['work_interfere'] = si.fit_transform(df[['work_interfere']]).ravel()

print('[INFO] Imputation complete.')
print('=' * 100)
