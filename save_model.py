# save_model.py

import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Pastikan folder model ada
os.makedirs('model', exist_ok=True)

# Load dataset
data = pd.read_csv('labeled_bitcoin_tweets.csv')

# Feature dan Label
X = data['clean_text']
y = data['label']

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Model training
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save vectorizer
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Vectorizer and Model saved successfully in folder 'model/'")
