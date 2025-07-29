import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load and merge Data
df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df[['text', 'label']]
df = df.sample(frac=1, random_state=42)  # Shuffle

# 2. Preprocess
stop_words = set(stopwords.words('english'))
def clean_text(text):
    return ' '.join([w for w in str(text).lower().split() if w.isalpha() and w not in stop_words])

df['text'] = df['text'].apply(clean_text)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. Evaluate
print("Test Accuracy:", model.score(X_test_vec, y_test))

# 7. Save Model & Vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
