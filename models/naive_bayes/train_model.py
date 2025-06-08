import pickle
import pandas as pd
import re
import string
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# --- Preprocessing & Training ---

# Download NLTK data (sekali saja)
nltk.download('punkt')

data = pd.read_csv("GojekAppReview_1.csv", encoding="latin-1")
data.drop(columns = ['userName', 'at', 'appVersion'], inplace = True)

# Batasi data (ambil 10.000 sampel pertama)
df = data.head(10000)

# START Prosessing
# 1. Clean the text
def remove_url(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = str(text).lower()
    text = remove_url(text)
    text = remove_emoji(text)
    text = re.sub(r'[^a-z\s]', ' ', text)  # hanya huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['content'].apply(lambda x: clean_text(x))


# 2. Adding aditional features -> Lenght of Review text, and percentage of punctuations in the Review text
def count_punct(text):
    text = str(text)
    count = sum([1 for char in text if char in string.punctuation])
    total_chars = len(text) - text.count(" ")
    return round(count / total_chars, 3) * 100 if total_chars > 0 else 0

df['content_len'] = df['content'].apply(lambda x: len(str(x)) - str(x).count(" "))

# 3. Normalisasi Kata Tidak Baku dalam Bahasa Indonesia
# Kata yang akan di normalisasi
normalisasi_kata = {
    "gk": "tidak",
    "ga": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "dr": "dari",
    "dgn": "dengan",
    "apk": "aplikasi",
    "yg": "yang"
}

def normalize_text(text):
    words = text.split()
    normalized = [normalisasi_kata[word] if word in normalisasi_kata else word for word in words]
    return " ".join(normalized)

df['normalized_text'] = df['cleaned_text'].apply(normalize_text)

# 4. Tokenization and Removing Stopwords
# Stopwords Bahasa Indonesia
stop_factory = StopWordRemoverFactory()
stopwords_ind = set(stop_factory.get_stop_words())

def tokenize_and_remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stopwords_ind]
    return filtered

df['filtered_tokens'] = df['normalized_text'].apply(tokenize_and_remove_stopwords)

# 5. Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokens(tokens):
    return " ".join([stemmer.stem(word) for word in tokens])

df['stemmed_text'] = df['filtered_tokens'].apply(stem_tokens)

# Labeling Sentimen (Negatif/Netral/Positif)
def label_sentiment(score):
    if score in [4.0, 5.0]:
        return 'positif'
    elif score == 3.0:
        return 'netral'
    else:
        return 'negatif'

df['label'] = df['score'].apply(label_sentiment)

# END Processing

# Feature Extraction (TF-IDF)
# Extract Feature With CountVectorizer -> Cleaning : convert all of data to lower case and removing all punctuation marks. 
X = df[['stemmed_text', 'content_len']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# ignore terms that occur in more than 50% documents and the ones that occur in less than 2
tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
tfidf_train = tfidf.fit_transform(X_train['stemmed_text'])
tfidf_test = tfidf.transform(X_test['stemmed_text'])

# PREDICTION
# Using Algoritma : Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
clf.score(tfidf_test, y_test)

# --- Save model and vectorizer ---
with open('models/naivebayes_model_nb.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('models/naivebayes_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model dan vectorizer berhasil disimpan.")