# Cell 4:
import os
import numpy as np
import tensorflow as tf
import os, random, re, json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Libraries loaded successfully.")
#------------------------------------------------------------------
# Cell 6:
GLOVE_PATH = "glove.6B.100d.txt"   
EMBED_DIM = 100
MAX_WORDS = 20000
MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 10

print("Parameters initialized.")
#------------------------------------------------------------------
# Cell 8:
df = pd.read_csv('sentimentdataset.csv')
df = df[['Text', 'Sentiment']]
df.head(5)

#------------------------------------------------------------------
# Cell 9:
class_counts = df['Sentiment'].value_counts()

print(class_counts)

#------------------------------------------------------------------
# Cell 10:
import matplotlib.pyplot as plt

label_counts = df['Sentiment']['Positive'].value_counts()

plt.figure(figsize=(6,4))
plt.bar(label_counts.index, label_counts.values, color=['red', 'green', 'gray'])
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

#------------------------------------------------------------------
# Cell 12:
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

def texts_to_padded(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

X_train = texts_to_padded(train_texts)
X_val = texts_to_padded(val_texts)
X_test = texts_to_padded(test_texts)

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

print(f"Tokenization complete. Example sequence length: {X_train.shape}")

#------------------------------------------------------------------
# Cell 14:
import zipfile, os, requests

url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip = "glove.6B.zip"

if not os.path.exists("glove.6B.100d.txt"):
    print("Downloading GloVe embeddings (862MB)... please wait 1–3 minutes.")
    r = requests.get(url, stream=True)
    with open(glove_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("Extracting GloVe files...")
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Done!")
else:
    print("GloVe file already exists.")

#------------------------------------------------------------------
# Cell 15:
def load_glove_embeddings(glove_path, word_index, embed_dim=100, max_words=20000):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            coefs = np.asarray(parts[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"Loaded {len(embeddings_index):,} GloVe word vectors.")

    num_words = min(max_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_dim))
    for word, i in word_index.items():
        if i < num_words:
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
    return embedding_matrix

embedding_matrix = load_glove_embeddings(GLOVE_PATH, tokenizer.word_index, EMBED_DIM, MAX_WORDS)

#------------------------------------------------------------------
# Cell 17:
from tensorflow.keras import layers, models, optimizers, callbacks

def build_model(num_words, embed_dim, max_len, embedding_matrix=None):
    inp = layers.Input(shape=(max_len,))
    if embedding_matrix is not None:
        emb = layers.Embedding(
            input_dim=num_words,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=False, 
            input_length=max_len
        )(inp)
    else:
        emb = layers.Embedding(num_words, embed_dim, input_length=max_len)(inp)
    
    x = layers.Bidirectional(layers.LSTM(64))(emb)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

num_words = min(MAX_WORDS, len(tokenizer.word_index) + 1)
model = build_model(num_words, EMBED_DIM, MAX_LEN, embedding_matrix)
model.summary()

#------------------------------------------------------------------
# Cell 19:
es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

print("Training complete.")

#------------------------------------------------------------------
# Cell 21:
y_prob_test = model.predict(X_test, batch_size=256).ravel()
brier_raw = brier_score_loss(y_test, y_prob_test)
auc_raw = roc_auc_score(y_test, y_prob_test)
print(f"Before calibration: Brier = {brier_raw:.4f}, AUC = {auc_raw:.4f}")

def reliability_diagram(y_true, y_prob, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    accuracies, confidences = [], []
    for i in range(bins):
        idx = bin_indices == i
        if np.any(idx):
            accuracies.append(np.mean(y_true[idx]))
            confidences.append(np.mean(y_prob[idx]))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(confidences, accuracies, 'o-', label='Model')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Accuracy')
    plt.title('Reliability Diagram (Before Calibration)')
    plt.legend()
    plt.grid(True)
    plt.show()

reliability_diagram(y_test, y_prob_test)

#------------------------------------------------------------------
# Cell 23:
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

print("Shapes:")
print("  p_val:", p_val.shape)
print("  logits_val:", logits_val.shape)
print("  y_val:", y_val.shape)
print("  p_test:", p_test.shape)
print("  logits_test:", logits_test.shape)
print("  y_test:", y_test.shape)

assert logits_val.shape[0] == y_val.shape[0], "val size mismatch"
assert logits_test.shape[0] == y_test.shape[0], "test size mismatch"


print("\nProb ranges:")
print("  p_val min/max:", p_val.min(), p_val.max())
print("  p_test min/max:", p_test.min(), p_test.max())

print("\nTemperature:", T_opt)
if not (0.01 < T_opt < 50):
    print("Temperature is unusual — may indicate problems (very small or very large).") 
p_test_raw = p_test
p_test_cal = probs_test_cal  
print("\nMetrics on TEST set:")
print(f"  Raw Brier:       {brier_score_loss(y_test, p_test_raw):.4f}")
print(f"  Calibrated Brier:{brier_score_loss(y_test, p_test_cal):.4f}")
print(f"  Raw AUC:         {roc_auc_score(y_test, p_test_raw):.4f}")
print(f"  Calibrated AUC:  {roc_auc_score(y_test, p_test_cal):.4f}")

def reliability_points(y, probs, bins=10):
    bins_edges = np.linspace(0,1,bins+1)
    bin_idx = np.digitize(probs, bins_edges) - 1
    confs, accs, counts = [], [], []
    for i in range(bins):
        idx = bin_idx == i
        if np.any(idx):
            confs.append(probs[idx].mean())
            accs.append(y[idx].mean())
            counts.append(idx.sum())
        else:
            confs.append(np.nan); accs.append(np.nan); counts.append(0)
    return np.array(confs), np.array(accs), np.array(counts)

confs_raw, accs_raw, counts_raw = reliability_points(y_test, p_test_raw)
confs_cal, accs_cal, counts_cal = reliability_points(y_test, p_test_cal)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot([0,1],[0,1],'k--', label='Perfect')
plt.plot(confs_raw, accs_raw, 'o-', label='Raw')
plt.title('Reliability - Raw')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot([0,1],[0,1],'k--', label='Perfect')
plt.plot(confs_cal, accs_cal, 'o-', label=f'Calibrated (T={T_opt:.3f})')
plt.title('Reliability - Calibrated')
plt.xlabel('Predicted probability')
plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()

#------------------------------------------------------------------
# Cell 25:
scaled_probs = 1 / (1 + np.exp(-logits_test / T_opt))
brier_scaled = brier_score_loss(y_test, scaled_probs)
auc_scaled = roc_auc_score(y_test, scaled_probs)

print(f"After calibration: Brier = {brier_scaled:.4f}, AUC = {auc_scaled:.4f}")

reliability_diagram(y_test, scaled_probs)

#------------------------------------------------------------------
# Cell 27:
SAVE_DIR = "sentiment_model_glove"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(SAVE_DIR, include_optimizer=False)

with open(os.path.join(SAVE_DIR, "tokenizer_meta.pkl"), "wb") as f:
    pickle.dump({
        "tokenizer": tokenizer,
        "max_len": MAX_LEN,
        "temperature": T_opt
    }, f)

print("Model and tokenizer saved successfully.")

#------------------------------------------------------------------
# Cell 29:
sample_texts = [
    "I absolutely loved this movie!",
    "The plot was boring and predictable.",
    "This product exceeded my expectations."
]

for text in sample_texts:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    raw_prob = model.predict(padded)[0,0]
    logit = np.log(raw_prob / (1 - raw_prob + 1e-8))
    scaled_prob = 1 / (1 + np.exp(-logit / T_opt))
    sentiment = "Positive" if scaled_prob >= 0.5 else "Negative"
    print(f"Text: {text}\n→ {sentiment} ({scaled_prob*100:.1f}%)\n")

#------------------------------------------------------------------
# Cell 30:


