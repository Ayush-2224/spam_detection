from django.db import models

# Create your models here.
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from django.conf import settings
from joblib import dump, load

class SpamDetector:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, 'models/spam_detector_model.joblib')
        self.vectorizer_path = os.path.join(settings.BASE_DIR, 'models/vectorizer.joblib')
        self.model = None
        self.vectorizer = None
        self._load_model()

    def _load_data(self, folder):
        data = []
        labels = []
        for filename in os.listdir(folder):
            label = 1 if 'spm' in filename else 0  # 1 for spam, 0 for non-spam
            with open(os.path.join(folder, filename), 'r') as file:
                data.append(file.read())
                labels.append(label)
        return pd.DataFrame({'text': data, 'label': labels})

    def _train_model(self):
        train_data = self._load_data(os.path.join(settings.BASE_DIR, 'models/train-mails'))
        X_train = self.vectorizer.fit_transform(train_data['text'])
        y_train = train_data['label']
        self.model.fit(X_train, y_train)

        # Save the model and vectorizer
        dump(self.model, self.model_path)
        dump(self.vectorizer, self.vectorizer_path)

    def _load_model(self):
        # Load model and vectorizer if they exist, else train a new one
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = load(self.model_path)
            self.vectorizer = load(self.vectorizer_path)
        else:
            self.model = MultinomialNB()
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self._train_model()

    def predict(self, text):
        X_input = self.vectorizer.transform([text])
        return self.model.predict(X_input)[0]
