import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#read csv
df = pd.read_csv('datasets/survey.csv')


#preprocessing

### dropping use less tables
df.drop(['comments', 'state', 'Timestamp', 'Country'], axis=1, inplace=True)

## removing null from target variable
df = df[df['treatment'].notna()]


## checking total null values
print("=========================================Total Null in dataset=========================================",df.isna().sum()) 


def cleanGender(g):
    g = str(g).strip().lower()
    if g in ['male', 'm', 'man', 'cis male', 'cis man']:
        return 'male'
    elif g in ['female', 'f', 'woman', 'cis female', 'femail']:
        return 'female'
    else:
        return 'other'

df['Gender'] = df['Gender'].apply(cleanGender)

# drop rows other then that
df = df[df['Gender'].isin(['male', 'female', 'other'])]

print("=========================================Total Genders=========================================",df['Gender'].value_counts())

## remove null with most frequent values
df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])
df['work_interfere'] = df['work_interfere'].fillna(df['work_interfere'].mode()[0])
