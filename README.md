# Sentiment-classification-using-RNN

## Project Overview

This notebook implements a sentiment classifier for movie reviews using **Long Short-Term Memory (LSTM)** networks and pre-trained **GloVe** word embeddings. The model is trained and evaluated on the IMDB dataset to predict binary sentiment (positive / negative). The goal is to demonstrate how sequence models and embeddings capture contextual information for sentiment prediction, and to compute standard evaluation metrics (accuracy, precision, recall, F1).

---

## Notebook

`Sentiment Classification Using RNNs.ipynb`

---

## Key Features

* Loads the IMDB dataset from `tensorflow.keras.datasets.imdb`.
* Preprocesses sequences with padding/truncation (fixed `maxlen`).
* Loads pre-trained **GloVe (50d)** embeddings and builds an embedding matrix for the vocabulary.
* Constructs an LSTM-based classifier:

  * Embedding layer (initialized with GloVe, not trainable)
  * LSTM (128 units)
  * Dense output with sigmoid activation
* Trains the model and visualizes training/validation loss and accuracy.
* Computes evaluation metrics: **accuracy**, **precision**, **recall**, **F1-score**.
* Example hyperparameters used in the notebook:

  * `vocab_size = 10000` (plus 3 special tokens)
  * `embedding_dim = 50`
  * `maxlen = 250`
  * `LSTM units = 128`
  * `epochs = 50`, `batch_size = 128`

---

## Results (from this run)

* Test accuracy: **0.8506**
* Precision: **0.8592**
* Recall: **0.8385**
* F1-score: **0.8487**

> Note: These values reflect one training run and can vary depending on random seeds, exact preprocessing, and whether the embedding layer is trainable.

---

## Files & Resources

* `6.1P - Sentiment Classification Using RNNs.ipynb` — main notebook
* `glove.6B.50d.txt` — GloVe embeddings file used to initialize embeddings (50-dimensional)
* The notebook relies on Keras' built-in IMDB dataset (downloaded automatically).

---

## How it works (high level)

1. **Load IMDB**: `keras.datasets.imdb.load_data(num_words=vocab_size, ...)`
2. **Build id→word mapping** and handle special tokens `[PAD],[START],[OOV]`.
3. **Pad sequences** to `maxlen` with `preprocessing.sequence.pad_sequences`.
4. **Load GloVe embeddings** into `embeddings_index`.
5. **Create `embedding_matrix`**: for each word in vocabulary, put pre-trained vector (zeros if missing).
6. **Define model**:

   ```python
   Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
   LSTM(128)
   Dense(1, activation='sigmoid')
   ```
7. **Compile & train** with `binary_crossentropy` and `adam`.
8. **Evaluate** on test set and compute precision/recall/F1.

---

## Quickstart — Run the notebook locally or in Colab

### Option A — Google Colab

1. Upload the notebook and the `glove.6B.50d.txt` file to your Drive (or change the notebook to download the GloVe zip automatically).
2. Open the notebook in Colab and run cells sequentially. Colab will download the IMDB dataset automatically.

### Option B — Local setup

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows PowerShell
pip install --upgrade pip
pip install tensorflow numpy matplotlib tqdm scikit-learn
# download GloVe (glove.6B.50d.txt) and place it next to the notebook
jupyter lab  # or jupyter notebook
```

Open the notebook and run cells. Update `data_path` for GloVe if necessary.

---

## Dependencies

Minimum (examples):

```
tensorflow (2.x)
numpy
matplotlib
tqdm
scikit-learn
```

If you want to use GPU acceleration, install TensorFlow GPU version matching your CUDA/cuDNN setup.

---

## Tips & Notes

* **Embedding trainability**: The notebook sets the embedding layer `trainable=False`. If you set it `True`, expect longer training but possibly improved performance.
* **Overfitting**: The training curves in the notebook indicate signs of overfitting (training acc >> val acc late in training). Use early stopping, dropout, or reduce epochs to mitigate.
* **Experimentation**: Try changing `embedding_dim`, LSTM size, making embeddings trainable, or using bidirectional LSTM for better performance.
* **Evaluation**: Use `sklearn.metrics` to compute precision, recall, and F1 as shown in the notebook.

---

## Suggested Experiments / Extensions

* Make embedding layer trainable or fine-tune GloVe vectors.
* Use 100d or 200d GloVe vectors for richer embeddings.
* Replace LSTM with BiLSTM, GRU, or Transformer encoder.
* Add dropout, recurrent dropout, or attention mechanism.
* Run k-fold cross-validation for robust performance estimates.
* Compare with simple baselines (Naive Bayes, FFNN) to quantify improvements.

---

## License & Attribution

* GloVe vectors: Stanford NLP Group (publicly available; check the GloVe license).
* IMDB dataset: Keras datasets (public).
* Add your preferred LICENSE file for this project if you plan to publish or share the code.

---

## Contact

If you want help adapting or extending this notebook, include your name/email here.

---

*This README introduces the code and experimentation in `6.1P - Sentiment Classification Using RNNs.ipynb`. Edit paths, hyperparameters, and notes to reflect your environment and goals.*
