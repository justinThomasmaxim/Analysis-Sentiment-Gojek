import pickle
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style("whitegrid")

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split, KFold

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# --- Preprocessing & Training ---
# Download NLTK data (sekali saja)
import nltk
nltk.download('stopwords')

df= pd.read_csv("GojekAppReview_1.csv", encoding="latin-1")
df.drop(columns=['userName', 'at', 'appVersion'], inplace=True)


# START Prosessing
# 3. Batasi data (ambil 10.000 sampel pertama)
df_sampled = df.head(10000)  # Ambil 10.000 baris pertama

# 4. Inisialisasi stemmer dan stopword remover Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopwords = StopWordRemoverFactory()
stopwords_sastrawi = set(factory_stopwords.get_stop_words())

# 5. Fungsi untuk pembersihan teks
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

# 6. Fungsi untuk tokenisasi
def tokenize(text):
    return text.split()

# 7. Fungsi untuk stemming dan hapus stopwords Sastrawi
def preprocess_text(text):
    tokens = tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_sastrawi]
    return ' '.join(tokens)

# 8. Terapkan pembersihan dan preprocessing pada data yang dibatasi (df_sampled)
df_sampled['cleaned_text'] = df_sampled['content'].apply(clean_text)
df_sampled['processed_text'] = df_sampled['cleaned_text'].apply(preprocess_text)

# 9. Labeling Multiclass: Positif, Netral, Negatif
def map_label(score):
    if score in [1.0, 2.0]:
        return 0  # Negatif
    elif score == 3.0:
        return 1  # Netral
    else:
        return 2  # Positif

df_sampled['label'] = df_sampled['score'].apply(map_label)

# 10. Split Data
X = df_sampled['processed_text'].values
y = df_sampled['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 11. Tokenisasi dan Padding
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# 12. Cross-validation 5-Fold untuk CNN
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

fold_no = 1
for train_idx, val_idx in kfold.split(X_train_pad):
    print(f"\n--- Training fold {fold_no} ---")

    # Membagi data menjadi fold
    X_train_fold, X_val_fold = X_train_pad[train_idx], X_train_pad[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Ubah label jadi one-hot encoding
    y_train_fold_cat = to_categorical(y_train_fold, num_classes=3)
    y_val_fold_cat = to_categorical(y_val_fold, num_classes=3)

    # Membuat model CNN
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=100, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 kelas
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Melatih model dengan data training fold
    model.fit(
        X_train_fold, y_train_fold_cat,
        epochs=10,
        batch_size=64,
        verbose=1,
        validation_data=(X_val_fold, y_val_fold_cat),
        callbacks=[early_stop]
    )

    # Evaluasi model pada fold
    scores = model.evaluate(X_val_fold, y_val_fold_cat, verbose=0)
    accuracies.append(scores[1])
    print(f"Fold {fold_no} accuracy: {scores[1]:.4f}")

    fold_no += 1

# --- Save model ---
with open('models/cnn_model_nb.pkl', 'wb') as f:
    pickle.dump(model, f)

# --- Save tokenizer ---
with open('models/cnn_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# --- Save text cleaning & preprocessing function using cloudpickle ---
import cloudpickle

# Save clean_text
with open('models/cnn_clean_text_fn.pkl', 'wb') as f:
    cloudpickle.dump(clean_text, f)

# Save preprocess_text (termasuk stemming & stopword)
with open('models/cnn_preprocess_text_fn.pkl', 'wb') as f:
    cloudpickle.dump(preprocess_text, f)

# --- Save padding configuration ---
pad_config = {
    'max_len': max_len,
    'padding': 'post',
    'truncating': 'post'
}

with open('models/cnn_pad_config.pkl', 'wb') as f:
    pickle.dump(pad_config, f)

print("Berhasil disimpan.")