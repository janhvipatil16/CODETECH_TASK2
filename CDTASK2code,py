# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD THE DATASET FROM CSV FILE
df = pd.read_csv("C:/Users/punit/OneDrive/Documents/IMDB Dataset.csv")
df.columns = ['text', 'label'] 
df['label'] = df['label'].map({'positive': 1, 'negative': 0})  

# SPLIT DATA INTO POSITIVE AND NEGATIVE SENTIMENT
neg = df[df['label'] == 0]
pos = df[df['label'] == 1]

# BALANCE THE DATASET BY RANDOM SAMPLING THE SMALLER CLASS TO MATCH THE SIZE OF THE LARGER CLASS
min_len = min(len(neg), len(pos))
neg = neg.sample(n=min_len, random_state=42)
pos = pos.sample(n=min_len, random_state=42)

# CONCATENATE THE BALANCED DATASET AND SHUFFLE THE ROWS
df = pd.concat([neg, pos]).sample(frac=1).reset_index(drop=True)

# FUNCTION TO CLEAN AND PREPROCESS THE TEXT DATA
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# APPLY THE CLEAN_TEXT FUNCTION TO PREPROCESS THE TEXT COLUMN
df['cleaned_text'] = df['text'].apply(clean_text)

# CONVERT THE TEXT DATA INTO NUMERICAL FEATURES USING TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(df['cleaned_text'])  
y = df['label']  

# SPLIT THE DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

# INITIALIZE AND TRAIN THE LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# MAKE PREDICTIONS ON THE TEST SET
y_pred = model.predict(X_test)

# PRINT CLASSIFICATION REPORT TO EVALUATE PERFORMANCE
print("Classification Report:")
print(classification_report(y_test, y_pred))

# PRINT CONFUSION MATRIX AND VISUALIZE USING A HEATMAP
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
