import json

with open('Sentiment Analysis Using Tensorflow (NLP Transfer Learning).ipynb', encoding='utf-8') as f:
    nb = json.load(f)

data_split_source = [
    "# Data Splitting\n",
    "texts = df['Text'].values\n",
    "# For binary classification: Map 'Positive' to 1, others to 0\n",
    "labels = (df['Sentiment'] == 'Positive').astype(int).values\n",
    "\n",
    "train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=SEED)\n",
    "train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=SEED)\n",
    "\n",
    "print(f'Train size: {len(train_texts)}')\n",
    "print(f'Val size: {len(val_texts)}')\n",
    "print(f'Test size: {len(test_texts)}')\n"
]
split_cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': data_split_source}

idx_10 = -1
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'label_counts = df' in ''.join(c.get('source', [])):
        idx_10 = i
        break

if idx_10 != -1:
    nb['cells'].insert(idx_10 + 1, split_cell)

idx_21 = -1
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'y_prob_test = model.predict' in ''.join(c.get('source', [])):
        idx_21 = i
        break

ts_source = [
    "from scipy.optimize import minimize\n",
    "from sklearn.metrics import log_loss\n",
    "\n",
    "def nll_fn(T, logits, y):\n",
    "    p = 1 / (1 + np.exp(-logits / T))\n",
    "    return log_loss(y, p)\n",
    "\n",
    "p_val = model.predict(X_val, batch_size=256).ravel()\n",
    "logits_val = np.log(p_val / (1 - p_val + 1e-8))\n",
    "\n",
    "p_test = model.predict(X_test, batch_size=256).ravel()\n",
    "logits_test = np.log(p_test / (1 - p_test + 1e-8))\n",
    "\n",
    "res = minimize(nll_fn, 1.0, args=(logits_val, y_val), method='L-BFGS-B', bounds=[(0.01, 50)])\n",
    "T_opt = res.x[0]\n",
    "print(f'Optimal Temperature: {T_opt:.4f}')\n",
    "\n",
    "probs_test_cal = 1 / (1 + np.exp(-logits_test / T_opt))\n"
]
ts_cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ts_source}

if idx_21 != -1:
    nb['cells'].insert(idx_21 + 2, ts_cell)

with open('Sentiment Analysis Using Tensorflow (NLP Transfer Learning).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
