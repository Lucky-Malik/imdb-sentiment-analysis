# 🎬 IMDb Sentiment Analysis using Deep Learning (Keras)

This project is a sentiment analysis model built using TensorFlow and Keras that classifies IMDb movie reviews as **Positive** or **Negative**. It uses a neural network with an embedding layer to handle natural language processing, and it is trained and evaluated on the IMDb dataset built into Keras.

---

## 📌 Project Overview

- **Goal**: Classify IMDb movie reviews as either positive or negative.
- **Dataset**: IMDb reviews from `keras.datasets.imdb`.
- **Model**: A simple feedforward neural network with an embedding layer.
- **Libraries**: TensorFlow/Keras, NumPy, Matplotlib, scikit-learn

---

## 🧠 Model Architecture

```
Input: Padded sequences of word indices (length = 500)
|
|--> Embedding layer (maps vocab indices to dense vectors of fixed size)
|--> Flatten layer (converts 2D embeddings into 1D for dense layers)
|--> Dense layer (64 units, ReLU activation, L2 regularization)
|--> Dropout (50% to prevent overfitting)
|--> Dense output layer (1 unit, Sigmoid activation for binary classification)
```

---

## 📁 File Structure

```
imdb-sentiment-analysis/
├── imdb_sentiment_analysis.py  # Main Python script
├── README.md                   # Project documentation
└── requirements.txt            # (optional) Python dependencies
```

---

## 🧪 Dataset Description

- The IMDb dataset contains **50,000** movie reviews:
  - **25,000** for training
  - **25,000** for testing
- Each review is already encoded as a sequence of integers (each representing a word).
- Reviews are labeled as:
  - **1**: Positive
  - **0**: Negative

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Lucky-Malik/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\Scripts\activate     # on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> If you don’t have `requirements.txt`, here are the key packages:
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### 4. Run the Script
```bash
python imdb_sentiment_analysis.py
```

---

## 📝 Code Explanation

```python
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
```

- Loads all necessary libraries for deep learning, visualization, and evaluation.

```python
vocab_size = 5000
maxlen = 500
```

- Limits the vocabulary to the top 5000 most frequent words.
- All reviews are padded/truncated to 500 words.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
```

- Loads and preprocesses the IMDb dataset.
- Pads sequences to ensure consistent input shape.

---

### 🔧 Model Building

```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

- **Embedding**: Converts word indices to dense vectors of size 32.
- **Flatten**: Converts the 2D embedding output to 1D.
- **Dense (ReLU)**: Fully connected layer with 64 neurons and L2 regularization.
- **Dropout**: Prevents overfitting by dropping 50% of the neurons.
- **Dense (Sigmoid)**: Output layer for binary classification (0 or 1).

---

### 🏁 Model Compilation & Training

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- Uses **Adam** optimizer.
- **Binary crossentropy** is ideal for binary classification tasks.

```python
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)
```

- Trains the model for 20 epochs.
- Uses 20% of the training data for validation.

---

### 🧪 Model Evaluation

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
```

- Evaluates the model on the test set and prints the accuracy.

---

### 📊 Visualization

```python
plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
...
```

- Plots **training vs validation accuracy** and **loss** to observe learning trends.

---

### 📈 Metrics and Performance

```python
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
```

- Predicts probabilities, then converts them to 0/1.
- Creates a **confusion matrix** to evaluate TP, TN, FP, FN.

```python
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
```

- Prints precision, recall, f1-score for each class.

---

## 🧾 Sample Output

```
Test Accuracy: 0.8652

Confusion Matrix:
[[10530  1970]
 [ 1420 11080]]

Classification Report:
              precision    recall  f1-score   support

    Negative       0.88      0.84      0.86     12500
    Positive       0.85      0.89      0.87     12500

    accuracy                           0.87     25000
```

---

## 💡 Future Improvements

- Use **LSTM or GRU** for better context capturing.
- Use **pre-trained embeddings** like GloVe.
- Implement **attention mechanisms**.

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify it!

---

## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [IMDb Dataset (Keras)](https://keras.io/api/datasets/imdb/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
