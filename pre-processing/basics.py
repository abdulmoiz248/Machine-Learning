import pandas as pd

print('[INFO] reading csv')
df=pd.read_csv('./datasets/survey.csv')

print('[INFO] Total Null Values')
print(df.isna().sum())
print("shape=",df.shape)

print('[INFO] PERCENTAGE OF NULL VALUES')
print('Percentage=', ((df.isna().sum().sum()/df.shape[0])*100))

print('[INFO] DROPPING STATE=',df.drop(columns=['state'],inplace=True))

print('[INFO] Applying mode on Self Employed table')
print(df['self_employed'].mode()[0])
df['self_employed'].fillna(df['self_employed'].mode()[0],inplace=True)
print(df['self_employed'].head())