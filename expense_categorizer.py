import pandas as pd
train_data = pd.read_csv("financial_transactions_train.csv")
print(train_data.head())
test_data = pd.read_csv("financial_transactions_test.csv")
X_train =train_data["Transaction_Text"]
y_train = train_data["Label"]
X_test = test_data["Transaction_Text"]
y_test = test_data["Label"]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy*100,"%")
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
new_transaction = input("Enter a new transaction: ")
new_transaction_tfidf = vectorizer.transform([new_transaction])
prediction = model.predict(new_transaction_tfidf)
print("Predicted Category:", prediction[0])
