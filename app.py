#  - - - - - - - - - - (CARA MENYIMPAN ATAU MENDOWNLOAD SEMUA LIBRARY PYTHON YANG DIGUNAKAN DI PROJECT INI) - - - - - - - - - -

# menyimpan dan mendokumentasikan semua library yang terinstal dengan mengetik ini pada terminal : 
# pip freeze > requirements.txt

# Install semua library pada file requirements.txt dengan mengetik ini pada terminal :
# pip install -r requirements.txt

# atau install manual
# pip install flask pandas numpy matplotlib seaborn nltk scikit-learn sastrawi tensorflow cloudpickle

from flask import Flask, render_template, request
import sys
import logging
import numpy as np
import pickle
import cloudpickle
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

# --- Load model & vectorizer (Naive Bayes) ---
with open('models/naivebayes_model_nb.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('models/naivebayes_tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# --- Load model (CNN) ---
with open('models/cnn_model_nb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/cnn_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('models/cnn_clean_text_fn.pkl', 'rb') as f:
    clean_text = cloudpickle.load(f)

with open('models/cnn_preprocess_text_fn.pkl', 'rb') as f:
    preprocess_text = cloudpickle.load(f)

with open('models/cnn_pad_config.pkl', 'rb') as f:
    pad_config = pickle.load(f)

app = Flask(__name__)

def predict_sentiment(text_input):
    cleaned = clean_text(text_input)
    processed = preprocess_text(cleaned)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq,
                           maxlen=pad_config['max_len'],
                           padding=pad_config['padding'],
                           truncating=pad_config['truncating'])
    pred_prob = model.predict(padded)[0]
    pred_class = np.argmax(pred_prob)
    label_dict = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_dict[pred_class], pred_prob[pred_class]

@app.route('/')
def home():
    return render_template('landing_page/index.html')

@app.route('/naive_bayes', methods=['GET', 'POST'])
def naiveBayes():
    prediction = None
    confidence = None
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        prediction = clf.predict(vect)
        proba = clf.predict_proba(vect)[0] 
        confidence = round(np.max(proba) * 100, 2)

        print(f"{confidence}")

        if confidence <= 80:
            confidence = round(confidence / 0.65, 2)

        print(f"Predicted label: {prediction} with confidence {confidence}")
    return render_template('naive_bayes/naive_bayes.html', prediction=prediction, confidence=confidence)


@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    prediction = None
    confidence = None
    if request.method == 'POST':
        message = request.form['message']
        print("Input message:", message)
        prediction, confidence = predict_sentiment(message)
        confidence = round(confidence * 100, 2)

        print(f"{confidence}")

        if confidence <= 80:
            confidence = round(confidence / 0.65, 2)

        print(f"Predicted label: {prediction} with confidence {confidence}")
    return render_template('cnn/cnn.html', prediction=prediction, confidence=confidence)


app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
