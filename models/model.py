import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
def load_data(folder):
    data = []
    labels = []
    for filename in os.listdir(folder):
        label = 1 if 'spm' in filename else 0  # 1 for spam, 0 for non-spam
        with open(os.path.join(folder, filename), 'r') as file:
            data.append(file.read())
            labels.append(label)
    return pd.DataFrame({'text': data, 'label': labels})

# Training and test data
train_data = load_data(r'D:\assignment_8\spam_detection\models\train-mails')
test_data = load_data(r'D:\assignment_8\spam_detection\models\test-mails')

# Preprocess data
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }

print(results)
