import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    return ' '.join([w for w in str(text).lower().split() if w.isalpha() and w not in stop_words])

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(vec).max()
    else:
        score = 1.0
    return pred, float(score)
