import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('datasets/spam.csv', encoding='latin-1')[['v1', 'v2']]
df = df[['v1', 'v2']].copy()
df.columns = ['label', 'message']

df['label']=df['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


vecotrizer=TfidfVectorizer(stop_words='english')
X_train_vec=vecotrizer.fit_transform(X_train)
X_test_vec=vecotrizer.transform(X_test)

model=SVC(kernel='linear',C=1)
model.fit(X_train_vec,y_train)

y_pred=model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

print("sample test =", model.predict(vecotrizer.transform(['free'])))

print(df.head())