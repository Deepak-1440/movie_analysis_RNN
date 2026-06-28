# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

A deep learning project that classifies movie reviews as **Positive** or **Negative** using a Simple Recurrent Neural Network (RNN) trained on the IMDB dataset.

---

## 📁 Project Structure

```
rnn_project/
├── imdb.ipynb        # Data loading, exploration & preprocessing
├── train.ipynb       # Model building, training & saving
├── predict.ipynb     # Sentiment prediction on custom reviews
├── app.py            # Streamlit web application
└── requirements.txt  # Project dependencies
```

---

## 🧠 Model Architecture

| Layer       | Type       | Details                       |
|-------------|------------|-------------------------------|
| Embedding   | Embedding  | vocab_size=10000, output=128  |
| SimpleRNN   | SimpleRNN  | units=128, activation=relu    |
| Dense       | Dense      | units=1, activation=sigmoid   |

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Callback:** EarlyStopping (patience=5, monitor=val_loss)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rnn_project.git
cd rnn_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `imdb.ipynb` | Load & explore the IMDB dataset |
| 2 | `train.ipynb` | Train the RNN model (saves `simple_rnn_imdb.h5`) |
| 3 | `predict.ipynb` | Run predictions on custom reviews |

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** [IMDB Dataset](https://keras.io/api/datasets/imdb/) via `tensorflow.keras.datasets`
- **Size:** 50,000 movie reviews (25,000 train / 25,000 test)
- **Labels:** Binary — `1` (Positive) / `0` (Negative)
- **Vocabulary size:** 10,000 most frequent words
- **Max sequence length:** 500

---

## 🖥️ Web App Preview

The Streamlit app allows you to:
- Enter any movie review in a text box
- Click **Classify** to get the sentiment
- View the prediction confidence score

---

## 📦 Requirements

```
tensorflow
numpy
pandas
scikit-learn
nltk
streamlit
joblib
matplotlib
seaborn
```

---

## 📝 Notes

- The trained model file `simple_rnn_imdb.h5` is **not included** in the repo (too large).  
  Run `train.ipynb` to generate it locally.
- To use a pre-trained model, place `simple_rnn_imdb.h5` in the project root directory.

---

## 👤 Author

**Deepak Singh**  
Feel free to fork, star ⭐, and contribute!
